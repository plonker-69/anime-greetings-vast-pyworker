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

import copy
import json
import os
import random

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# --- handler.py's local server (env-overridable to match start_model_server.sh) ---
MODEL_SERVER_URL  = os.environ.get("PYWORKER_MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("PYWORKER_MODEL_SERVER_PORT", "8000"))

# Number of GPUs this box serves, one handler.py + one ComfyUI each, behind
# router.py on MODEL_SERVER_PORT. Set by vast-entrypoint.sh from `nvidia-smi
# -L`, so one image serves the 1x/2x/4x workergroups with no rebuild.
NUM_GPU_WORKERS = int(os.environ.get("NUM_GPU_WORKERS", "1"))
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
    # NOTE, 2026-08-29: "CUDA out of memory" and "torch.cuda.OutOfMemoryError"
    # were removed from this list. They were the last two BARE substrings here
    # -- i.e. the last place worker health was inferred from whatever text
    # ComfyUI happened to print, which is the same mistake that cost a healthy
    # worker on the KJNodes traceback above.
    #
    # Both were wrong in the common case. ComfyUI's model_management CATCHES
    # OOM, frees models and retries, so a job that prints the string can still
    # succeed -- and the worker would have been destroyed anyway. An oversized
    # single job that genuinely OOMs should fail that job and leave the warm
    # worker available for the next request, not take the box down.
    #
    # OOM is now classified in handler.py (_failure_is_worker_fatal): it calls
    # ComfyUI's /free, then asks /system_stats whether ComfyUI is still
    # serving. Only if it is not does it print the sentinel below. Health is
    # decided from ComfyUI's API, and reaches this list as one explicit,
    # unambiguous signal.
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


# --- TTS-Audio-Suite CosyVoice3: two node packs, ONE copy of the weights ---
# The 11 shared files + llm.pt that TTS-Audio-Suite verifies before it decides
# whether to download. Read off the pinned commit (127bfe32,
# engines/cosyvoice/cosyvoice_downloader.py: SHARED_FILES / _verify_model),
# not guessed -- if this list drifts from theirs the check below silently
# stops protecting anything.
TTS_SUITE_REQUIRED = (
    "cosyvoice3.yaml", "campplus.onnx", "flow.pt", "hift.pt",
    "speech_tokenizer_v3.onnx", "llm.pt",
    "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt", "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json", "CosyVoice-BlankEN/vocab.json",
)


def link_tts_audio_suite_model(models_root):
    """Point models/TTS/CosyVoice/Fun-CosyVoice3-0.5B at the CosyVoice3 copy
    the boot fetch already pulled for comfyui_fl-cosyvoice3
    (models/cosyvoice/), so TTS-Audio-Suite's CosyVoiceEngineNode resolves
    without downloading a second ~5.4GB copy of the same HuggingFace repo.

    WHY THIS LIVES HERE, of all places: this file is the ONE artefact on the
    worker that deploys by `git push` to PYWORKER_REPO rather than by a
    ~41GB docker build -- start_server.sh re-clones it at every container
    boot. fetch-models.py holds the same function and is where it belongs,
    but that is in the image, so on the currently-live tag the link does not
    exist. Calling it here makes the merged TTS+I2V+upscale graph
    (workflow_api_full.json) runnable on the live image today, and keeps
    working afterwards as a no-op once the image carries the link itself.

    Runs at import, which is before the startup benchmark -- i.e. before any
    job, including the one the benchmark itself sends. TTS-Audio-Suite
    resolves the model path at NODE EXECUTION time, not at ComfyUI startup,
    so a link created after ComfyUI has already booted is still picked up.

    Best-effort, never raises: a missing link costs a slow first job, which
    is not a reason to fail a worker boot. Keep in step with the copies in
    ../../fetch-models.py, ../../download-models.py and
    ../../download-models-runtime.py.
    """
    import os
    from pathlib import Path

    source = Path(models_root) / "cosyvoice" / "Fun-CosyVoice3-0.5B"
    target = Path(models_root) / "TTS" / "CosyVoice" / "Fun-CosyVoice3-0.5B"
    try:
        if target.is_symlink() or target.exists():
            where = os.readlink(target) if target.is_symlink() else "real directory"
            print(f"[tts-link] {target} already present ({where}) -- leaving it alone")
            return
        missing = [f for f in TTS_SUITE_REQUIRED if not (source / f).exists()]
        if missing:
            print(f"[tts-link] NOT linking: {source} is missing {missing} -- "
                  "TTS-Audio-Suite will self-download its own copy on first use")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(source, target.parent), target)
        print(f"[tts-link] {target} -> {os.path.relpath(source, target.parent)} "
              "(TTS-Audio-Suite now reuses the fl-cosyvoice3 weights)")
    except Exception as e:
        print(f"[tts-link] WARNING: could not link the TTS-Audio-Suite model copy: {e}")

# Runs at import: before the benchmark, therefore before any job.
link_tts_audio_suite_model(
    os.path.join(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"), "models"))

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

# Node ids of the two seeded samplers in the benchmark workflow -- mirrors
# worker/workflow.py's NODE["SAMPLER"] / NODE["TTS"]; keep in sync if the
# graph changes (both templates are kept in lockstep, see that file's
# module docstring).
_BENCHMARK_SAMPLER_NODE = "128"  # WanVideoSampler
# There is no TTS node any more. CosyVoice was retired on 2026-09-15 and speech
# comes from the Fish Speech S2-Pro instance, so benchmark_payload.json is the
# real-audio graph and _BENCHMARK_TTS_NODE ("318", FL_CosyVoice3_CrossLingual)
# is gone with it. Seeding a node that no longer exists would KeyError on every
# benchmark, i.e. on every worker boot.


def _make_benchmark_payload() -> dict:
    """Fresh copy of the benchmark request with a new random sampler seed,
    every call.

    HANDOFF.md bug this fixes ("Fix the Vast benchmark before it serves
    real traffic"): ComfyUI caches node outputs by input hash within the
    same running process, and `do_warmup=True` below submits this payload
    TWICE on boot -- once to force the model into VRAM (warmup), once as
    the "measured" run Vast's autoscaler uses to size this worker's
    throughput. With a static payload (the old `dataset=[_benchmark_request]`
    below) the second submission is byte-for-byte identical to the first,
    so WanVideoSampler just replays its cached output instead of
    generating -- the measured run clocked in ~41x
    faster than a real request, and the autoscaler never scaled the fleet
    up under load.

    Passing BenchmarkConfig a `generator` instead of a `dataset` gets this
    function called fresh for every submission, warmup included (see
    vastai's GenericApiPayload.for_test(): `dataset` is drawn from via
    random.choice -- a fixed pool that can and does repeat -- `generator`
    is invoked new each time). Randomizing the sampler seed on every call
    guarantees no two submissions in one boot share a seed, so the
    "measured" run always does a real, cache-miss generation -- matching
    how worker/workflow.py's build_workflow() already randomizes seeds for
    every real order.
    """
    payload = copy.deepcopy(_benchmark_request)
    workflow = payload["input"]["workflow"]
    workflow[_BENCHMARK_SAMPLER_NODE]["inputs"]["seed"] = random.randrange(2**31)
    return payload

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
            # False on a 1-GPU box (unchanged behaviour: requests FIFO-queue
            # in the PyWorker), True when router.py has more than one backend
            # to hand them to. The comment above describes the single-GPU
            # reason this was False -- concurrent write_input_files() calls
            # clobbering each other, and one ComfyUI that cannot take two
            # jobs. Both are now handled a layer down: each handler owns its
            # OWN ComfyUI, its OWN input/output dirs and its OWN flock, and
            # router.py hands a job only to a backend that is idle. 2026-08-30.
            allow_parallel_requests=NUM_GPU_WORKERS > 1,
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
                generator=_make_benchmark_payload,
                runs=1,
                # Must match the number of backends, and this is what makes
                # a multi-GPU box report HONEST capacity.
                # backend.__run_benchmark computes
                #   max_throughput = total_workload / time_elapsed
                # so N concurrent benchmark jobs at workload 10000 each,
                # finishing in T, report N*10000/T instead of 10000/T. The
                # autoscaler therefore learns the box is worth N workers by
                # MEASUREMENT, with no hand-set multiplier to drift out of
                # sync -- which is exactly how the cost/workload_calculator
                # mismatch happened (see the note on workload_calculator
                # below). Leave the two coupled.
                #
                # Note this only takes effect with allow_parallel_requests
                # True; while it is False, vastai forces benchmark
                # concurrency to 1 regardless of this value.
                #
                # Cold start becomes 2N generations (do_warmup) but they run
                # in parallel, so wall-clock is roughly unchanged.
                concurrency=NUM_GPU_WORKERS,
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
            #
            # ⚠️ CONTRACT, 2026-08-27 -- this number is HALF of a pair. Every
            # client that submits to this endpoint must declare the SAME
            # number as its /route/ `cost`:
            #   ../../smoketest.py            -> VAST_REQUEST_COST
            #   ../../../worker/adapters.py   -> VastBackend(cost=...)
            #   testing/run_i2v_batch.py      -> VAST_REQUEST_COST
            # The two halves are one currency. The startup benchmark reports
            # this worker's throughput to the autoscaler as
            #   max_perf = (this number) / (seconds per generation)
            # i.e. 10000 / ~174s ~= 57 perf units/s. `cost` is the only signal
            # telling the autoscaler how much work an arriving request is
            # ("the estimated compute resources for the request", per Vast's
            # /route/ API reference). Mismatch them and the endpoint's
            # utilisation and queue-time maths are wrong by exactly that
            # ratio: with cost=100 against this 10000, a real 266s backlog
            # scored as ~1.7s of work, ~1% utilisation against target_util
            # 0.9, and the endpoint sat flat at one worker forever. That was
            # the "won't scale past 1 worker" bug -- see HANDOFF.md 2026-08-27.
            #
            # Vast's own reference workers pair these explicitly (tgi/openai:
            # cost=max_tokens vs workload_calculator=max_tokens; comfyui-json:
            # COST=100 vs a 100.0 calculator). Their wan example -- the source
            # of this 10000.0 -- omits cost client-side and therefore ships
            # the same 100x mismatch. Don't copy its client.
            #
            # Changing this value invalidates any cached benchmark score:
            # backend.__run_benchmark writes max_perf to `.has_benchmark` in
            # $SERVER_DIR (/workspace/vast-pyworker) and short-circuits on it,
            # so an instance restarted in place keeps the OLD units. Recruit
            # fresh instances (or delete that file) rather than restarting.
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
