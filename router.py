#!/usr/bin/env python3
"""Fan-out proxy: one PyWorker-facing port, N handler.py backends.

WHY THIS EXISTS
---------------
Vast's PyWorker forwards every request to exactly ONE address --
`WorkerConfig(model_server_url, model_server_port)`, 127.0.0.1:8000 for us.
On a multi-GPU instance we want one handler.py + one ComfyUI pinned per GPU,
so something has to sit on :8000 and hand each arriving job to whichever
backend is free. That is all this does.

Deliberately NOT a load balancer. There is no routing policy beyond "give it
to a free one", because every job on this box costs the same and takes the
same GPU. Backends are handed out from a queue, so a job blocks until one is
genuinely idle rather than piling two jobs onto one ComfyUI -- which is
exactly the double-dispatch crash handler.py's own flock exists to prevent
(HANDOFF.md, 2026-08-26).

DESIGN NOTES
------------
- **Stdlib only.** No aiohttp/requests import. This process starts before
  anything has verified the image's Python environment, and a missing
  dependency here would take the whole worker down for no good reason.
- **Health-gated.** A backend only enters the free queue once its own
  healthcheck answers, so a job can never land on a ComfyUI that is still
  loading 40GB of weights.
- **No request timeout on the forward.** Generations run 200-950s and the
  PyWorker itself uses `ClientTimeout(total=None)` for the same reason. A
  timeout here would abort a perfectly healthy long clip -- the same class of
  bug as the client-side idle-TCP reap resolved on 2026-08-28.
- **Prints the ready sentinel.** worker.py's MODEL_LOAD_LOG_MSG watches for
  "[handler] ComfyUI is up". On a multi-GPU box the individual handlers each
  print their own per-port ready line (which no longer matches), and THIS
  process prints the sentinel once ALL backends are up. Without that, Vast
  would mark a 4x box ready as soon as the first GPU warmed and route jobs to
  three ComfyUI instances that were still booting.

This runs for N=1 as well. One code path, no single-GPU special case.
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_HOST = os.environ.get("ROUTER_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("ROUTER_PORT", "8000"))
# Backends are contiguous from this port: 8001, 8002, ... 8000+N
BACKEND_BASE_PORT = int(os.environ.get("ROUTER_BACKEND_BASE_PORT", "8001"))
NUM_BACKENDS = int(os.environ.get("NUM_GPU_WORKERS", "1"))
BACKEND_HOST = os.environ.get("ROUTER_BACKEND_HOST", "127.0.0.1")

# How long to wait for every backend to come up before giving up on the ones
# that haven't. Generous: a cold ComfyUI boot includes a model fetch.
BACKEND_READY_TIMEOUT_S = int(os.environ.get("ROUTER_READY_TIMEOUT", "3600"))
BACKEND_POLL_INTERVAL_S = float(os.environ.get("ROUTER_POLL_INTERVAL", "5.0"))
# How long a request waits for ANY backend to come free before giving up with a
# 503. Normal queueing behind a long generation is expected and fine -- this is
# only here so that "every backend is quarantined" fails visibly instead of
# hanging a connection forever. Matches HandlerConfig.max_queue_time.
QUEUE_TIMEOUT_S = float(os.environ.get("ROUTER_QUEUE_TIMEOUT", "3600"))

BACKENDS = [f"http://{BACKEND_HOST}:{BACKEND_BASE_PORT + i}" for i in range(NUM_BACKENDS)]

# Readiness marker each handler writes once ITS ComfyUI answers. Must match
# handler.py's COMFY_READY_FILE; vast-entrypoint.sh sets both explicitly.
#
# ⚠️ Do NOT go back to using the HTTP probe for readiness. handler.py starts its
# RunPod API server immediately and loads ComfyUI in a background thread (~54s),
# so the port answers at t=0 while ComfyUI is still loading weights. Probing it
# marks every backend ready instantly and defeats the entire reason this router
# owns the readiness sentinel. The HTTP probe is still correct for LIVENESS
# (is this process answering at all) -- that is _backend_alive below.
READY_FILE_PATTERN = os.environ.get("ROUTER_READY_FILE_PATTERN", "/tmp/comfy-ready.{port}")
COMFY_BASE_PORT = int(os.environ.get("ROUTER_COMFY_BASE_PORT", "8188"))
READY_FILES = [READY_FILE_PATTERN.format(port=COMFY_BASE_PORT + i, i=i)
               for i in range(NUM_BACKENDS)]

# Free backends. A blocking get() is the whole concurrency control: at most
# NUM_BACKENDS jobs are ever in flight, and a job waits rather than doubling up.
_free = queue.Queue()
_ready_count = 0
_ready_lock = threading.Lock()

# Per-backend tallies. These exist to ANSWER a question we currently guess at:
# do single GPUs die on boxes whose other GPUs stay healthy? If they do,
# draining a dead backend (instead of letting handler.py's FATAL_WORKER condemn
# the whole instance) becomes worth building. If they don't -- and most
# FATAL_WORKER causes are image-level, so all N backends fail together -- then
# condemning is correct and this stays a counter. Do not design the drain
# before these numbers say something. 2026-08-30.
_stats = {b: {"ok": 0, "fail": 0, "quarantined": 0} for b in BACKENDS}
_stats_lock = threading.Lock()

# In-flight jobs, backend -> start time. Exists purely for observability: a
# generation runs 200-950s during which the router would otherwise print
# NOTHING, which is indistinguishable from being hung. That ambiguity cost a
# real debugging session on 2026-08-30 -- a worker sat quiet after readiness
# and there was no way to tell "benchmark running" from "stuck". The heartbeat
# below turns that silence into visible progress. 2026-08-30.
_inflight = {}
_inflight_lock = threading.Lock()
HEARTBEAT_S = float(os.environ.get("ROUTER_HEARTBEAT", "60"))


def _heartbeat():
    while True:
        time.sleep(HEARTBEAT_S)
        with _inflight_lock:
            snap = [(b, time.time() - t0) for b, t0 in _inflight.items()]
        if snap:
            detail = ", ".join(f"{b.rsplit(':', 1)[-1]}={age:.0f}s" for b, age in snap)
            _log(f"heartbeat: {len(snap)}/{NUM_BACKENDS} busy, running for {detail}")


def _bump(backend, key):
    with _stats_lock:
        _stats[backend][key] += 1
        return _stats[backend][key]


def _log(msg):
    print(f"[router] {msg}", flush=True)


def _backend_alive(base):
    """RunPod's --rp_serve_api exposes GET / (its Swagger page). Same probe
    worker.py uses against this router."""
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            return 200 <= r.status < 500
    except urllib.error.HTTPError:
        # Answering at all -- even 404 -- proves the HTTP server is up.
        return True
    except Exception:
        return False


def _backend_ready(base, ready_file):
    """Ready = ComfyUI is genuinely up AND the handler's API answers.

    Both halves are needed: the file alone could be stale from a previous boot
    (handler.py removes it before each relaunch, but belt and braces), and the
    HTTP probe alone is true ~54s too early. See READY_FILE_PATTERN above.
    """
    return os.path.exists(ready_file) and _backend_alive(base)


def _wait_for_backend(base, ready_file):
    global _ready_count
    deadline = time.time() + BACKEND_READY_TIMEOUT_S
    while time.time() < deadline:
        if _backend_ready(base, ready_file):
            _free.put(base)
            with _ready_lock:
                _ready_count += 1
                n = _ready_count
            _log(f"backend {base} ready ({n}/{NUM_BACKENDS})")
            if n == NUM_BACKENDS:
                # The sentinel worker.py's MODEL_LOAD_LOG_MSG watches for.
                # Printed only when EVERY GPU is warm.
                print("[handler] ComfyUI is up", flush=True)
                _log(f"all {NUM_BACKENDS} backend(s) ready")
            return
        time.sleep(BACKEND_POLL_INTERVAL_S)
    _log(f"WARNING: backend {base} never came up within "
         f"{BACKEND_READY_TIMEOUT_S}s (ready file {ready_file}) "
         f"-- serving without it")


def _return_or_quarantine(backend):
    """Put a backend back in rotation only if it still answers.

    Without this, a dead backend goes straight back into the free queue and
    grabs the very next job -- failing it in milliseconds while healthy
    backends are busy, so it preferentially eats the queue. handler.py's
    FATAL_WORKER does condemn the box, but Vast takes time to act on it, and
    every job burned in that window is a real customer's order.

    A quarantined backend is re-probed in the background and rejoins the pool
    if it recovers (e.g. ComfyUI came back on its own after an OOM).
    """
    if _backend_alive(backend):
        _free.put(backend)
        return
    n = _bump(backend, "quarantined")
    _log(f"WARNING: backend {backend} not answering -- quarantined "
         f"(quarantine #{n}); re-probing every {BACKEND_POLL_INTERVAL_S}s")

    def _reprobe():
        while True:
            time.sleep(BACKEND_POLL_INTERVAL_S)
            if _backend_alive(backend):
                _log(f"backend {backend} answering again -- back in rotation")
                _free.put(backend)
                return

    threading.Thread(target=_reprobe, daemon=True).start()


class Router(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_request(self, code="-", size="-"):
        """Suppress the per-request access log for successful requests.

        PyWorker polls MODEL_HEALTHCHECK_ENDPOINT ("/") continuously, so the
        default access log floods handler.log with `"GET / HTTP/1.1" 200 -`
        forever. That file is what worker.py tails AND what reaches
        `vastai logs`, so the flood buries the lines that matter -- job
        dispatch, the heartbeat, backend failures. Observed in production
        2026-08-30.

        Everything worth seeing is already logged explicitly by do_POST and
        the heartbeat, so success needs no access-log line at all. Failures
        still get one -- those are never noise.
        """
        try:
            if int(code) >= 400:
                _log(f'{self.requestline} -> {code}')
        except (TypeError, ValueError):
            pass

    def log_message(self, fmt, *args):
        # Still reached by log_error() for genuine server-side problems.
        _log(fmt % args)

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # worker.py's MODEL_HEALTHCHECK_ENDPOINT. 200 once at least one
        # backend can actually serve; 503 before that, so nothing upstream
        # mistakes "process started" for "can take work".
        with _ready_lock:
            ready = _ready_count
        with _stats_lock:
            stats = {b: dict(v) for b, v in _stats.items()}
        body = json.dumps({"ready": ready, "of": NUM_BACKENDS, "backends": stats}).encode()
        self._send(200 if ready else 503, body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length)
        _log(f"received {self.path} ({length}B); {_free.qsize()}/{NUM_BACKENDS} backend(s) free")

        # Blocks until a backend is idle. This IS the queue.
        try:
            backend = _free.get(timeout=QUEUE_TIMEOUT_S)
        except queue.Empty:
            _log(f"ERROR: no backend free within {QUEUE_TIMEOUT_S}s "
                 f"-- all {NUM_BACKENDS} busy or quarantined")
            self._send(503, json.dumps(
                {"error": "router: no backend available"}).encode())
            return
        t0 = time.time()
        with _inflight_lock:
            _inflight[backend] = t0
        _log(f"dispatched {self.path} -> {backend} (generation can run 200-950s; "
             f"heartbeat every {HEARTBEAT_S:.0f}s)")
        # Set BEFORE the try: every exit path below (success, HTTPError,
        # connection failure) reaches the finally, and an unset name there is
        # an UnboundLocalError that breaks the request it was meant to report.
        failed = False
        try:
            req = urllib.request.Request(
                backend + self.path,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # No timeout on purpose -- see the module docstring.
            with urllib.request.urlopen(req) as r:
                body = r.read()
                code = r.status
        except urllib.error.HTTPError as e:
            # The BACKEND answered, with an error status. handler.py returns a
            # failed job as HTTP 200 with an {"error": ...} body anyway, so
            # this is an unexpected-but-alive backend, not a dead one. Do not
            # quarantine it.
            body = e.read()
            code = e.code
        except Exception as e:
            # A dead backend must not take the request down silently. Report
            # it as a job failure; WORKER health is handler.py's call, not
            # ours, and it signals that with its own FATAL_WORKER sentinel.
            n = _bump(backend, "fail")
            _log(f"ERROR: backend {backend} failed ({n} failure(s) so far): {e}")
            body = json.dumps({"error": f"router: backend {backend} failed: {e}"}).encode()
            code = 502
            failed = True
        finally:
            with _inflight_lock:
                _inflight.pop(backend, None)
            if failed:
                _return_or_quarantine(backend)
            else:
                _bump(backend, "ok")
                _free.put(backend)
        with _stats_lock:
            st = dict(_stats[backend])
        _log(f"{self.path} -> {backend} [{code}] {time.time() - t0:.1f}s "
             f"(ok={st['ok']} fail={st['fail']} quarantined={st['quarantined']})")
        self._send(code, body)


def main():
    _log(f"routing {LISTEN_HOST}:{LISTEN_PORT} -> {NUM_BACKENDS} backend(s): "
         f"{', '.join(BACKENDS)}")
    _log(f"readiness gated on: {', '.join(READY_FILES)}")
    for b, rf in zip(BACKENDS, READY_FILES):
        threading.Thread(target=_wait_for_backend, args=(b, rf), daemon=True).start()
    threading.Thread(target=_heartbeat, daemon=True).start()
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Router).serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[handler] FATAL_WORKER: router failed to start: {e}", flush=True)
        sys.exit(1)
