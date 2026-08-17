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

echo "[vast-entrypoint] starting handler.py in --rp_serve_api mode"
# Note: output is duplicated to both the log file AND this container's own
# stdout (via process substitution + tee, not a plain pipe -- a plain pipe
# would make $! below capture tee's PID instead of handler.py's, breaking
# the immediate-crash check right after this). This is deliberate: worker.py
# only relays specific whitelisted log-line patterns from handler.log into
# `vastai logs` (its own "Info from model logs:" / "Got log line indicating
# error:" wrapper), which silently swallows anything else -- including a
# real Python traceback's actual stack frames, past the first "Traceback
# (most recent call last):" line. Tee-ing straight to stdout means
# `vastai logs <instance_id>` shows the raw, complete output no matter what
# worker.py does or doesn't forward -- see HANDOFF.md, 2026-08-16.
python3 -u /handler.py \
    --rp_serve_api \
    --rp_api_host 127.0.0.1 \
    --rp_api_port 8000 \
    > >(tee -a "$LOG_DIR/handler.log") 2>&1 &
HANDLER_PID=$!
echo "[vast-entrypoint] handler.py pid=$HANDLER_PID, logging to $LOG_DIR/handler.log"

# If handler.py's API server dies outright (crash before ComfyUI even
# starts, e.g. a port conflict), fail fast and loudly instead of leaving
# worker.py to wait forever for a "[handler] ComfyUI is up" line that will
# never come.
sleep 2
if ! kill -0 "$HANDLER_PID" 2>/dev/null; then
    echo "[vast-entrypoint] FATAL: handler.py exited immediately -- see $LOG_DIR/handler.log" >&2
    tail -n 100 "$LOG_DIR/handler.log" >&2 || true
    exit 1
fi

echo "[vast-entrypoint] handing off to start_server.sh (worker.py bootstrap)"
exec /start_server.sh
