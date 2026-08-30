#!/bin/bash
# Container ENTRYPOINT for Vast Serverless deployment of the greeting
# pipeline. Set as ENTRYPOINT (not CMD) by Dockerfile.vast-serverless,
# which deliberately OVERRIDES whatever ENTRYPOINT BASE_IMAGE declared,
# rather than running as CMD underneath it. Read on for why.
#
# History: the first version of this file was meant to run as CMD,
# trusting BASE_IMAGE's own ENTRYPOINT to `exec` into it. That's NOT safe
# across this project's Dockerfile variants -- checked all four
# (2026-08-15, see HANDOFF.md): `Dockerfile`/`Dockerfile.blackwell`'s
# `reassemble-models.sh` does end with a clean `exec "$@"`, but
# `Dockerfile.lightweight`/`Dockerfile.lightweight-blackwell`'s
# `fetch-models-entrypoint.sh` does not -- it pipes CMD's output through
# `log-timestamps.py` instead (`"$@" 2>&1 | python3 -u /log-timestamps.py`),
# so CMD would run as a pipeline child, not a true PID-1 replacement.
# Docker's own comment on that file already flagged this as a gap for "a
# long-lived production worker" -- exactly what a Vast Serverless PyWorker
# is, and per this project's own recent validation runs (see HANDOFF.md,
# 2026-08-15: `anime-greetings-vast-lite-blackwell:v4`), the fetch-at-boot
# lightweight lines look to be the ones actually in active use now, not
# the baked-model lines -- so "just use the baked-model line for Vast"
# wasn't a real fix, it was dodging the actually-relevant image line.
#
# What this script does about it: owns the whole boot sequence itself, as
# the real ENTRYPOINT, instead of trusting BASE_IMAGE's ENTRYPOINT+CMD
# chain at all --
#   1. Run whichever pre-step BASE_IMAGE actually needs, auto-detected by
#      which script is present on disk (not a build-arg, so this works
#      unmodified no matter which BASE_IMAGE was chosen):
#      `reassemble-models.sh` (chunk reassembly + checksum verification,
#      baked-model lines) or `fetch-models-entrypoint.sh` (HuggingFace
#      fetch, lightweight lines). Called directly with NO arguments, not
#      `exec`'d -- verified (2026-08-15, see HANDOFF.md) that both
#      scripts' own trailing "$@"-based command line becomes a harmless
#      no-op when called with zero arguments (`exec "$@"` is a documented
#      bash no-op with nothing to exec; `"$@" 2>&1 | python3 -u
#      /log-timestamps.py` becomes an empty command piped into a script
#      that reads stdin line-by-line and exits cleanly on immediate EOF --
#      checked `log-timestamps.py` directly, `for line in sys.stdin:`
#      over zero lines just falls through). So this runs each script's
#      real side effect (reassembly, or the HF fetch) without inheriting
#      whatever that script would otherwise have done with a real CMD.
#   2. Start handler.py in RunPod's *local hosted-API* mode
#      (--rp_serve_api), which serves a plain HTTP /runsync route on
#      127.0.0.1:8000 -- confirmed locally that this mode answers
#      GET / -> 200 and POST /runsync -> 200 (even on internal handler
#      errors, which come back as a FAILED-status JSON body, not an HTTP
#      error -- see handler.py's own comment on its except block).
#   3. exec into Vast's own pyworker bootstrap (start_server.sh, vendored
#      alongside this file from https://github.com/vast-ai/pyworker),
#      which clones PYWORKER_REPO (wherever this folder -- worker.py +
#      requirements.txt + benchmark_payload.json -- ends up pushed to),
#      installs its deps, and runs worker.py in the foreground. This is
#      now the FIRST `exec` in the whole boot chain, so it really does
#      replace PID 1 -- not a pipeline child three scripts deep the way
#      the lightweight lines' original chain would have made it.
#
# Vendored, not fetched live: start_server.sh is copied into this image at
# build time rather than curl'd from GitHub at every cold start, so an
# autoscaled worker's boot doesn't depend on GitHub's availability and
# doesn't silently pick up upstream changes mid-fleet. Re-sync by hand from
# https://github.com/vast-ai/pyworker/blob/main/start_server.sh periodically.

set -euo pipefail

# Boot forensics FIRST, before the model fetch, so its output frames every
# later line: was this a container restart or a recreate, did the kernel
# OOM-kill anything in this cgroup last time, and what did memory look like in
# the seconds before the previous run died. Best-effort by design -- guarded
# with `|| true` under `set -e` so a diagnostic can never be the reason a
# worker fails to boot. See boot-forensics.sh's header and HANDOFF.md.
#
# NOTE 2026-08-29: these are TWO DIFFERENT FAULTS, not one -- an earlier
# version of this comment said "one bug" and that linkage is withdrawn.
#   - 2026-08-24 long-clip death: RESOLVED. Client-side idle TCP connection
#     reap, fixed with OS keepalives; 56 clips have since completed, longest
#     953.6s. Nothing here diagnoses it because there is nothing left to
#     diagnose.
#   - 2026-08-27 container restart: still UNEXPLAINED. Proven Docker-level
#     by a PID reset (422 -> 59, so a new PID namespace). No leading cause;
#     host RAM is neither established nor excluded. This is what the
#     forensics below are actually for.
# Both calls are plain SUBPROCESSES on purpose, not `source`: errexit is not
# inherited across a fork+exec, so nothing in the diagnostic can trip this
# script's own `set -e`. The sampler still outlives boot-forensics.sh -- it is
# backgrounded, orphaned when that script exits, and reparented to whatever
# becomes PID 1 (start_server.sh, via the exec at the end of this file).
if [ -x /boot-forensics.sh ]; then
    /boot-forensics.sh boot || true
    /boot-forensics.sh sampler || true
else
    echo "[vast-entrypoint] boot-forensics.sh not present on this image -- skipping"
fi

if [ -x /reassemble-models.sh ]; then
    echo "[vast-entrypoint] baked-model image detected -- running reassemble-models.sh"
    /reassemble-models.sh
elif [ -x /fetch-models-entrypoint.sh ]; then
    echo "[vast-entrypoint] fetch-at-boot image detected -- running fetch-models-entrypoint.sh"
    /fetch-models-entrypoint.sh
else
    echo "[vast-entrypoint] WARNING: neither /reassemble-models.sh nor /fetch-models-entrypoint.sh found on this image -- assuming BASE_IMAGE handles its own models some other way (e.g. Dockerfile.sageattention-test, not a real deploy target)" >&2
fi

LOG_DIR="/var/log/vast-pyworker"
mkdir -p "$LOG_DIR"
# ⚠️ worker.py tails ONE file -- MODEL_LOG_FILE, hardcoded to
# $LOG_DIR/handler.log -- and that single stream is how it learns the worker is
# ready (MODEL_LOAD_LOG_MSG) and how it detects a dead worker
# (MODEL_ERROR_LOG_MSGS / FATAL_WORKER). EVERY process whose output worker.py
# must see has to append to THIS file, whatever else it also writes to.
#
# Broke exactly this on 2026-08-30: the multi-GPU change gave each handler its
# own handler.$i.log and the router router.log, so nothing wrote handler.log at
# all. ComfyUI booted fine, the router printed the readiness sentinel to
# router.log, worker.py saw an empty file, never marked the model loaded, never
# ran the benchmark, and the worker hung silently forever. Per-process logs are
# for humans; handler.log is the contract.
COMBINED_LOG="$LOG_DIR/handler.log"
touch "$COMBINED_LOG"

# --- One handler.py + one ComfyUI per GPU, behind router.py -----------------
#
# NUM_GPU_WORKERS is auto-detected from the box, so ONE image serves the 1x,
# 2x and 4x workergroups with no rebuild and no per-group tag. Override with
# the env var to under-subscribe a box deliberately.
#
# ⚠️ The binding constraint is SYSTEM RAM, not GPU count. The workflow keeps
# the transformer in CPU memory on purpose (node 122 load_device:
# offload_device), so each ComfyUI holds roughly 20-30GB of host RAM. Size the
# workergroup's `cpu_ram` filter accordingly -- the 1x group's `cpu_ram>=64`
# is sized for ONE. Getting this wrong looks exactly like the unexplained
# container restarts (HANDOFF.md, 2026-08-27), so do not confuse the two.
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
if [ -z "$NUM_GPU_WORKERS" ] || [ "$NUM_GPU_WORKERS" -lt 1 ] 2>/dev/null; then
    echo "[vast-entrypoint] WARNING: could not detect GPUs -- assuming 1"
    NUM_GPU_WORKERS=1
fi
export NUM_GPU_WORKERS
echo "[vast-entrypoint] launching $NUM_GPU_WORKERS handler(s), one per GPU"

# Output is duplicated to both the log file AND this container's own stdout
# (via process substitution + tee, not a plain pipe -- a plain pipe would make
# $! capture tee's PID instead of the handler's, breaking the immediate-crash
# check below). Deliberate: worker.py only relays specific whitelisted log-line
# patterns from handler.log into `vastai logs`, silently swallowing everything
# else -- including a real traceback's stack frames past the first line.
# Tee-ing straight to stdout means `vastai logs <instance_id>` shows the raw,
# complete output regardless -- see HANDOFF.md, 2026-08-16.
HANDLER_PIDS=""
i=0
while [ "$i" -lt "$NUM_GPU_WORKERS" ]; do
    # Each handler owns exactly one GPU, one ComfyUI port, and its own input/
    # output dirs. The dirs are NOT cosmetic: write_input_files() writes the
    # incoming portrait/voice under names the workflow hardcodes, so a shared
    # input/ would let two concurrent jobs overwrite each other's portrait and
    # deliver a video of the wrong person with no error raised.
    CUDA_VISIBLE_DEVICES="$i"     COMFY_PORT="$((8188 + i))"     COMFY_READY_FILE="/tmp/comfy-ready.$((8188 + i))"     COMFY_INPUT_DIR="/tmp/comfy-input-$i"     COMFY_OUTPUT_DIR="/tmp/comfy-output-$i"     python3 -u /handler.py         --rp_serve_api         --rp_api_host 127.0.0.1         --rp_api_port "$((8001 + i))"         > >(tee -a "$COMBINED_LOG" "$LOG_DIR/handler.$i.log") 2>&1 &
    HANDLER_PIDS="$HANDLER_PIDS $!"
    echo "[vast-entrypoint] handler[$i] pid=$! gpu=$i comfy_port=$((8188 + i)) api_port=$((8001 + i)) log=$LOG_DIR/handler.$i.log"
    i=$((i + 1))
done

# router.py owns :8000 -- the single address PyWorker forwards to
# (WorkerConfig model_server_port). It hands each job to whichever handler is
# free, and prints the "[handler] ComfyUI is up" readiness sentinel only once
# EVERY backend has written its /tmp/comfy-ready.<port> marker. NOT once the
# API ports answer -- those answer at t=0 while ComfyUI is still loading
# weights (~54s), which would mark the box ready far too early. It runs for
# NUM_GPU_WORKERS=1 too: one code path, no single-GPU special case.
# Clear stale readiness markers before the router starts looking for them.
# handler.py also removes its own before each (re)launch; this covers a
# container whose /tmp survived from an earlier boot.
rm -f /tmp/comfy-ready.* 2>/dev/null || true
python3 -u /router.py > >(tee -a "$COMBINED_LOG" "$LOG_DIR/router.log") 2>&1 &
ROUTER_PID=$!
echo "[vast-entrypoint] router.py pid=$ROUTER_PID on :8000, logging to $LOG_DIR/router.log"

# Fail fast and loudly if anything died on the spot (a port conflict, a bad
# import), instead of leaving worker.py waiting forever for a readiness line
# that will never come.
sleep 2
for p in $HANDLER_PIDS $ROUTER_PID; do
    if ! kill -0 "$p" 2>/dev/null; then
        echo "[vast-entrypoint] FATAL: pid $p exited immediately -- see $LOG_DIR/" >&2
        tail -n 100 "$LOG_DIR"/handler.*.log "$LOG_DIR"/router.log 2>/dev/null >&2 || true
        exit 1
    fi
done

echo "[vast-entrypoint] handing off to start_server.sh (worker.py bootstrap)"
exec /start_server.sh
