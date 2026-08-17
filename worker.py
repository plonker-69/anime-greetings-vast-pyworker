"""Vast Serverless PyWorker for the InfiniteTalk + CosyVoice greeting pipeline.

This does NOT reimplement job execution -- it proxies to our existing
handler.py, run locally in RunPod's own "hosted API" mode
(`handler.py --rp_serve_api`), which stands up a plain HTTP server with a
`/runsync` route (POST in, blocking, JSON out). That route already does
everything a from-scratch Vast worker would need to do: boot ComfyUI from
the network volume, queue the workflow, poll to completion, collect the
mp4, and return it -- plus every fix that's landed in handler.py itself
(--cache-classic, the silent-video audio fallback, LATEST.mp4, HTTPError
body surfacing; see ../../HANDOFF.md). We deliberately did NOT adopt
ai-dock/comfyui-api-wrapper (the middleware Vast's own comfyui-json/wan
examples sit in front of) -- it's a third-party dependency we've never
run, and handler.py already covers the same ground.

Deployment shape (see README.md in this folder for the full walkthrough):
  1. The container boots ComfyUI's *proxy*, not ComfyUI directly: something
     on the box (a provisioning/onstart script -- see start_model_server.sh)
     runs `python3 -u handler.py --rp_serve_api --rp_api_host 127.0.0.1
     --rp_api_port 8000 >> /var/log/vast-pyworker/handler.log 2>&1 &`
     handler.py itself starts ComfyUI as its own subprocess on first
     request, so nothing else needs to launch ComfyUI separately.
  2. Vast's serverless template startup script clones PYWORKER_REPO (this
     folder) and runs `python worker.py` -- that's THIS file.
  3. This worker tails the same log file for the
     "[handler] ComfyUI is up" line handler.py already prints (see
     handler.py's start_comfy()), runs one real benchmark generation
     against /runsync to confirm the whole pipeline actually works, and
     then proxies production traffic to the same route.

Known gap, by design: RunPod's local hosted-API server
(runpod.serverless.modules.rp_fastapi) exposes no /health-style route --
only /run, /runsync, /stream/{id}, /status/{id} (all POST) plus a GET "/"
that serves its Swagger docs page. We use that GET "/" as the PyWorker
healthcheck: it's cheap and always 200 while the FastAPI process is alive,
but it only proves handler.py's HTTP layer is up, NOT that ComfyUI is
still healthy underneath it. A ComfyUI crash *after* a successful boot
(e.g. mid-run CUDA OOM) won't flip the healthcheck -- it'll only show up
as the next real request failing. Confirmed locally (no ComfyUI installed
in that sandbox) that `handler.py --rp_serve_api` serves GET / -> 200 and
POST /runsync -> HTTP 200 with a JSON {"status": "FAILED", "error": ...}
body even when the handler itself raises -- so a "successful" proxy round
trip and a "successful" video generation are two different things; the
benchmark step below is what actually proves generation works, once, at
boot.
"""

import json
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# --- handler.py's local server (env-overridable to match start_model_server.sh) ---
MODEL_SERVER_URL  = os.environ.get("PYWORKER_MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("PYWORKER_MODEL_SERVER_PORT", "8000"))
MODEL_LOG_FILE    = os.environ.get(
    "PYWORKER_MODEL_LOG_FILE", "/var/log/vast-pyworker/handler.log"
)

# GET "/" is RunPod's Swagger docs page in --rp_serve_api mode -- the only
# GET route it exposes, and the closest thing to a healthcheck available.
# See the module docstring for what this does and doesn't verify.
MODEL_HEALTHCHECK_ENDPOINT = "/"

# handler.py prints this once ComfyUI's own HTTP server answers
# /system_stats (see start_comfy() in handler.py) -- NOT once models are
# actually loaded into VRAM. That first real load happens during the
# benchmark's warmup call below.
MODEL_LOAD_LOG_MSG = [
    "[handler] ComfyUI is up",
]

# handler.py's except block now prints "[handler] ERROR: job failed: ..."
# on any failure (added 2026-08-15, see handler.py) -- it used to only
# return the error in the HTTP body, invisible to a log watcher, which is
# exactly the gap local testing surfaced: RunPod's --rp_serve_api mode
# answers HTTP 200 even on a failed job, so vastai's benchmark (which only
# checks HTTP status) couldn't tell a real success from handler.py's own
# soft failure. This pattern alone also makes the two ComfyUI-boot-failure
# patterns below reachable for the first time (their RuntimeError text
# flows through the same print via str(e)) -- previously they could only
# ever match text ComfyUI's own subprocess happened to print, never these.
#
# Important nuance, confirmed by re-running the same local test with this
# fix in place (see HANDOFF.md's 2026-08-15 Vast Serverless section for
# the full before/after log): this does NOT prevent the worker from
# briefly marking itself ready during a failing startup benchmark.
# vastai's log tailer (backend.py's tail_log()) awaits each log line's
# handler sequentially, and the ModelLoaded case's benchmark call is
# awaited from inside that same handler -- so the tailer is blocked on
# the very HTTP call whose failure we want it to notice, and can't read
# this new line until that call (and the readiness decision) already
# resolved. What IS confirmed: the tailer catches up immediately
# afterward -- in the retest, both queued "[handler] ERROR" lines (one
# per benchmark call: the warmup plus the runs=1 measured run) were read
# and flipped the backend to an errored state (metrics._model_errored,
# which persists error_msg for Vast's status reporting) in the SAME
# second "marking model as loaded" printed. So the false-ready window
# measured here was ~1s, not indefinite, and any failure during real
# production traffic (or a later cold boot) is caught immediately, same
# as normal -- the gap is specifically the startup benchmark's own brief
# window, not a standing blind spot.
MODEL_ERROR_LOG_MSGS = [
    # NOTE, 2026-08-16: this list used to also include
    # "[handler] ERROR: job failed", "Traceback (most recent call last):",
    # and "Value not in list: ". All three were too broad -- they match
    # request-level/recoverable failures (a single bad job, a workflow
    # validation error) and, worse, ComfyUI's own benign, caught-and-
    # continued custom-node-import tracebacks (confirmed: KJNodes' MiniMax
    # nodes failing to import with a non-fatal ModuleNotFoundError was
    # enough to trip "Traceback (most recent call last):" and get a
    # perfectly healthy worker destroyed by Vast). This list should only
    # contain genuinely unrecoverable *worker*-health conditions -- a bad
    # job should fail and return an error, leaving the warm worker
    # available for the next request, not get the whole box destroyed.
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    # Explicit, deliberate sentinel handler.py prints from BOTH start_comfy()
    # call sites -- _boot_comfy_or_die (the eager initial-boot thread) and
    # handler(job)'s wrapped restart-on-request call -- whenever start_comfy()
    # itself raises. Superseded "ComfyUI did not come up within"/"ComfyUI
    # exited with code" as of 2026-08-16: those were substring-matching
    # exception text wherever it happened to leak into the log (the initial
    # boot's uncaught-thread-exception dump, or falling through to the
    # generic "[handler] ERROR: job failed" print on a restart failure).
    # Now handler.py classifies "ComfyUI itself is broken" explicitly at
    # both call sites instead of worker.py inferring it from message text,
    # so both old patterns are fully covered by this one sentinel.
    "[handler] FATAL_WORKER:",
]

MODEL_INFO_LOG_MSGS = [
    "[handler] starting ComfyUI",
    "[handler] queued prompt",
    "[handler] WARNING",
]

# Real production request: one portrait + one voice clip -> InfiniteTalk +
# CosyVoice video. Used both as the "does this box actually work" boot
# check and as the throughput sample -- same tradeoff Vast's own Wan 2.2
# pyworker example makes (workers/wan/worker.py benchmarks a real ~20-step
# generation, not a synthetic fast one). There's no fast/cheap workflow
# that would exercise the same nodes, and a benchmark that doesn't
# actually run the pipeline wouldn't catch the model-loading and node
# failures that have bitten this project before.
_BENCHMARK_PAYLOAD_PATH = os.path.join(os.path.dirname(__file__), "benchmark_payload.json")
with open(_BENCHMARK_PAYLOAD_PATH) as _f:
    _benchmark_request = json.load(_f)  # {"input": {"workflow": ..., "files": [...]}}
benchmark_dataset = [_benchmark_request]

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/runsync",
            # ComfyUI (and handler.py's own start_comfy/queue_prompt/
            # wait_for_completion, which are plain synchronous calls
            # against ONE ComfyUI instance) cannot handle two jobs at
            # once -- concurrent write_input_files() calls alone would
            # clobber each other. Requests FIFO-queue in the PyWorker
            # instead of running in parallel.
            allow_parallel_requests=False,
            # Governs admission, not a per-request timer: PyWorker rejects
            # (HTTP 429) new requests once *estimated* wait time -- queued
            # workload / measured throughput -- exceeds this. Each job
            # here takes minutes, not seconds, so the wan example's 10.0
            # would reject almost everything queued behind an in-flight
            # job. 3600s gives room for a handful of jobs to queue behind
            # one long-running generation before PyWorker starts shedding
            # load.
            max_queue_time=3600.0,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
                runs=1,
                # allow_parallel_requests=False above forces the
                # benchmark's own concurrency to 1 regardless of this
                # value (see vastai's backend.__run_benchmark) -- set
                # explicitly anyway so the intent survives an SDK version
                # bump.
                concurrency=1,
                # True (default) = one warmup generation (real model load
                # into VRAM, slow) + one measured generation, i.e. TWO
                # full video generations before this worker is marked
                # ready. Matches Vast's own Wan pyworker pattern and gives
                # an honest throughput number. Flip to False to halve
                # cold-start time if the measured throughput isn't worth
                # it for this workload's volume.
                do_warmup=True,
            ),
            # Flat per-job cost, same reasoning as the Wan example
            # (workers/wan/worker.py uses a flat 10000.0): this pipeline's
            # cost doesn't scale with an easily-read request field the way
            # LLM token counts do.
            workload_calculator=lambda _: 10000.0,
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

Worker(worker_config).run()
