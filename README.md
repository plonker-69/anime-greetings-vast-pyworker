# Vast Serverless PyWorker for the greeting pipeline

This folder is the Vast Serverless integration for the InfiniteTalk +
CosyVoice greeting-video pipeline. It does not reimplement job execution --
it wraps the existing `../../handler.py` (already baked into every
`vast/Dockerfile*` variant) so it can run as a Vast Serverless worker
instead of (or alongside) a RunPod one.

## Why this shape, not the ai-dock middleware

Vast's own reference workers (`comfyui-json`, `wan`) sit in front of
[ai-dock/comfyui-api-wrapper](https://github.com/ai-dock/comfyui-api-wrapper)
-- a separate FastAPI middleware that boots ComfyUI, queues a prompt, polls
to completion, and exposes `/generate/sync`. We don't adopt it: `handler.py`
already does every one of those steps itself, plus every real fix that's
landed in it (`--cache-classic`, the silent-video audio fallback,
`LATEST.mp4`, ComfyUI HTTPError body surfacing -- see `../../HANDOFF.md`)
that a fresh copy of a third-party wrapper wouldn't have and we've never
run in production.

`handler.py --rp_serve_api` (RunPod's own local hosted-API mode) already
gives us the same shape as `/generate/sync`: a plain HTTP server with one
blocking, synchronous route (`/runsync`) that takes a job in, runs it, and
returns the result. Confirmed locally (see worker.py's docstring for the
exact test): `GET /` -> 200, `POST /runsync` -> HTTP 200 with a
`{"status": "FAILED", "error": ...}` body even when the handler raises
internally -- i.e. handler.py never needs to know it's being proxied by
something Vast-shaped instead of RunPod's own queue poller.

## Files in this folder

| File | Purpose |
|---|---|
| `worker.py` | The actual PyWorker. Built on `vastai`'s `Worker`/`WorkerConfig` (see below for how that was verified). |
| `requirements.txt` | `worker.py`'s own deps -- just `vastai`. Installed by `start_server.sh` into its own venv, separate from ComfyUI's Python env. |
| `benchmark_payload.json` | Copy of `../../smoke_test_payload_5090.json` -- a real portrait+voice request. Doubles as the worker's boot-time correctness check and its throughput sample. |
| `start_server.sh` | Vendored verbatim from `vast-ai/pyworker` (not written by us -- see "What this actually does" below). Clones `PYWORKER_REPO`, installs `requirements.txt`, runs `worker.py`. |
| `vast-entrypoint.sh` | New, ours. Container `ENTRYPOINT` for Vast: runs the right model pre-step, starts `handler.py --rp_serve_api` in the background, then hands off to `start_server.sh`. |
| `boot-forensics.sh` | New, ours. Boot counter, cgroup OOM-kill counter and a background memory sampler, run from `vast-entrypoint.sh`. Makes a silent container death diagnosable without SSH -- see "Boot forensics" below. |
| `Dockerfile.vast-serverless` | New, ours. **A direct, self-contained fork of `../../Dockerfile.lightweight-blackwell`** -- same base image, same SageAttention compile, same custom-node list and pins, same HF-fetch-at-boot model strategy, copied line-for-line, with only the final `ENTRYPOINT`/`CMD` lines changed to run `vast-entrypoint.sh` instead. Not a wrapper around a separately-built image (see Setup steps §2 for why that changed 2026-08-15). Not baked in: `worker.py`/`requirements.txt`/`benchmark_payload.json` -- those are cloned at container boot from `PYWORKER_REPO`, not at image build time. |

## How the pieces fit together, end to end

```
container boot
  -> ENTRYPOINT vast-entrypoint.sh        (Dockerfile.vast-serverless's own
                                            ENTRYPOINT/CMD -- this file is a
                                            self-contained fork, there's no
                                            separate base image to inherit
                                            from anymore, see Setup steps §2)
       -> runs fetch-models-entrypoint.sh, with no arguments (this fork is
          the lightweight-blackwell line, so it's always this script, not
          reassemble-models.sh -- see vast-entrypoint.sh's own comments for
          why calling it with no args is still safe and still does the real
          HuggingFace fetch)
       -> handler.py --rp_serve_api &     (background; logs -> /var/log/vast-pyworker/handler.log)
       -> exec start_server.sh            (vendored from vast-ai/pyworker)
            -> git clone $PYWORKER_REPO into $WORKSPACE_DIR/vast-pyworker
            -> pip install -r requirements.txt   (just `vastai`)
            -> python3 -m worker              == worker.py in this folder
                 -> tails handler.log for "[handler] ComfyUI is up"
                 -> runs one real benchmark generation against /runsync
                    (this is what actually loads ComfyUI's models into VRAM
                    for the first time -- "ComfyUI is up" only means its
                    HTTP server answered /system_stats, not that models
                    are loaded)
                 -> marks the worker ready, starts proxying /runsync traffic
                 -> also now watches for "[handler] ERROR: job failed" in
                    the same log (added 2026-08-15 after local testing
                    showed a failed benchmark could otherwise still read
                    as "ready" -- see HANDOFF.md for the measured ~1s
                    false-ready window this leaves, and why it wasn't
                    worth chasing further)
```

`handler.py` itself starts ComfyUI as its own subprocess on first request
(see `start_comfy()`) -- nothing here launches ComfyUI separately.

## Setup steps

### 1. Push this folder as `PYWORKER_REPO`

`vastai`'s pyworker bootstrap expects `worker.py` and `requirements.txt` at
the *root* of a git repository (`vast-ai/pyworker`'s own README: "Put
`worker.py` and `requirements.txt` at the root of a public Git repository"
-- confirmed against the live docs). Push this folder's contents (all five
files) to its own repo, or a repo where they sit at the root. Then, on the
Vast Serverless endpoint/template config, set:

- `PYWORKER_REPO` = that repo's URL
- `PYWORKER_REF` = a branch/tag/commit (optional; defaults to the repo's default branch)

These are plain environment variables on the template/workergroup config
(same "Environment Variables" section as any other Docker env var) -- Vast's
docs don't expose a dedicated UI field for them, they're just env vars
`start_server.sh` reads.

### 2. Build the image

`Dockerfile.vast-serverless`, in this same folder, is a **direct,
self-contained fork of `../../Dockerfile.lightweight-blackwell`** -- not a
wrapper around a separately-built image. Two earlier versions of this file
tried a `FROM $BASE_IMAGE` build-arg wrapper instead (first trusting the
base image's own `ENTRYPOINT`, then overriding it) -- corrected 2026-08-15
after review: every other `Dockerfile*` in this project is a self-contained
fork of its nearest relative (see the top of `Dockerfile.lightweight-
blackwell` itself for that same convention), and a wrapper that depends on
first building and pushing a prerequisite image as a separate manual step
broke that pattern for no real benefit. So this file now copies
`Dockerfile.lightweight-blackwell`'s content line-for-line -- base image,
ffmpeg, ComfyUI core clone, SageAttention pre-flight and compile (Blackwell
arch, same pins), all seven custom node clones with their pinned commits,
ComfyUI-Manager offline config, the HF-fetch dependencies -- and changes
only the final section: instead of `ENTRYPOINT ["/fetch-models-entrypoint.sh"]`
/ `CMD ["python", "-u", "/handler.py"]`, it adds `start_server.sh` and
`vast-entrypoint.sh` and sets `ENTRYPOINT ["/vast-entrypoint.sh"]` / `CMD []`.
Keep everything above that final section in sync BY HAND if
`Dockerfile.lightweight-blackwell` changes -- same rule that file already
applies to its own two parents.

Build it (from `vast/`, same convention as every other `Dockerfile.*`
variant, but note the `-f` path is one level down since this file lives in
`serverless/vast_pyworker/`):

```bash
docker build --platform linux/amd64 \
  -f serverless/vast_pyworker/Dockerfile.vast-serverless \
  -t anime-greetings-vast-serverless:v1 \
  .
```

No `--build-arg` needed -- this is now a complete, standalone image build,
same as `docker build -f Dockerfile.lightweight-blackwell ...` for the
RunPod line. Needs a rented Blackwell card (RTX 5090 / B100 / B200) to
verify, same GPU constraint as its parent -- do NOT deploy this to
Ampere/Ada/Hopper.

**Why `ENTRYPOINT` is set, not just `CMD`:** `vast-entrypoint.sh` needs to
own the whole boot sequence -- run the model pre-step, launch `handler.py`,
then hand off to `start_server.sh` -- rather than being appended after
some other script's own bootstrap. Setting both `ENTRYPOINT
["/vast-entrypoint.sh"]` and `CMD []` makes that explicit and guarantees
`vast-entrypoint.sh` is PID 1, with `exec /start_server.sh` at the end of
it becoming the one real hand-off to the pyworker process.

**A real, unresolved risk worth reading before deploying this** (found
2026-08-15 while forking this file, embedded in the Dockerfile's own
header comments too): `../../auto-fetch-models.sh` exists because Vast has
been confirmed, 5 boxes in a row on 2026-08-14, to NOT reliably run a
custom Docker image's own `ENTRYPOINT` when deploying it on Vast's regular
on-demand rental flow -- `auto-fetch-models.sh` works around it by firing
from `/etc/profile.d` on SSH login instead, which has nothing to hook into
on a Vast Serverless worker (no SSH, no login shell).

**Update, 2026-08-15, after actually reading Vast's own docs** (not just
inferring from a code comment -- see HANDOFF.md for the full citation
list): the finding is explained, in Vast's own words.
[docs.vast.ai/instances/launch-modes](https://docs.vast.ai/instances/launch-modes)
and [template-settings](https://docs.vast.ai/documentation/templates/template-settings)
both confirm regular instances have a per-template Launch Mode: "Docker
ENTRYPOINT" mode runs the image exactly as built; "SSH"/"Jupyter" modes
state outright that "the docker entrypoint for your image will not be
run -- it will be replaced with our instance setup script." That's
documented, intentional platform behavior, not an undocumented quirk --
the 5-boxes-in-a-row finding is almost certainly boxes that were on SSH
or Jupyter launch mode (the default/common choice for an interactive
rental), not a platform-wide inability to run ENTRYPOINT.

For Serverless specifically,
[creating-new-pyworkers](https://docs.vast.ai/guides/serverless/creating-new-pyworkers.md)
describes the worker boot sequence as "the start-server script (provided
by the template)" doing exactly four things -- clone `PYWORKER_REPO`,
install `requirements.txt`, start the model server, run `worker.py` --
which is precisely what `vast-entrypoint.sh` + the vendored
`start_server.sh` already do here. That page never mentions ENTRYPOINT,
CMD, or Launch Mode at all (checked verbatim, not just summarized) --
Serverless templates don't appear to expose the Jupyter/SSH/ENTRYPOINT
selector regular instances have, consistent with (though not an explicit
statement of) Serverless always running the image as declared. This is
now well-supported by primary sources, not just plausible inference --
but no page states outright "Serverless always honors ENTRYPOINT," so
**the first thing to check on any real Serverless deploy of this image is
still whether `vast-entrypoint.sh`'s first log line
(`[vast-entrypoint] fetch-at-boot image detected...`) ever appears.** If
it doesn't, this whole ENTRYPOINT-based design needs a different
mechanism, before spending more time on anything downstream of it.

Also found, not yet acted on: Vast documents a newer, Python-SDK-based
Serverless deployment path (`app.image()`/`.run_script()`/
`app.ensure_ready()`, see
[deployments/configuration](https://docs.vast.ai/guides/serverless/deployments/configuration.md))
that sidesteps hand-written Dockerfiles entirely. This project still uses
the classic `PYWORKER_REPO` + template-env-var path `vast-ai/pyworker`'s
own README documents -- that path is still current and documented, just
not the only one Vast now offers. Worth a closer look at some point, not
investigated further here.

This intentionally does NOT require rebasing onto `vastai/base-image` or
adopting its Supervisor-based service model (confirmed by inspecting that
repo directly: it's a general-purpose interactive-instance base image with
Jupyter/desktop/portal machinery, and Vast's own docs don't say Serverless
requires it -- `vast-ai/pyworker`'s own README describes only "the
template's startup script" running `start_server.sh`, with no base-image
requirement). If a future need comes up for long-running non-serverless
Vast instances of this pipeline, that's the base to reach for; not needed
here.

### 3. Set the usual Vast platform env vars

`start_server.sh` hard-requires `CONTAINER_ID` (fails fast via
`report_error_and_exit` otherwise) -- this should already be set
automatically by Vast's platform on every instance, the same way RunPod
sets `RUNPOD_POD_ID`, but **worth confirming on the first real deploy**
rather than assuming. Everything else it reads (`REPORT_ADDR`, `USE_SSL`,
`WORKER_PORT`, `WORKSPACE_DIR`) has a sane default and doesn't need to be
set explicitly.

`handler.py`'s own existing env vars (`COMFY_ROOT`, `COMFY_JOB_TIMEOUT`,
`BUCKET_ENDPOINT_URL`, etc.) are unaffected -- `vast-entrypoint.sh` launches
it exactly as today's RunPod CMD does, just with `--rp_serve_api` added.

## Boot forensics (`boot-forensics.sh`)

Serverless workers can't be SSH'd into -- no `dmesg`, no `free`, no
`nvidia-smi` at the moment of a failure -- and when a container dies and
restarts, `vastai logs` is reset, so the evidence for what killed it is
destroyed by the thing that killed it. That's the standing blocker on the
unexplained-container-restart investigation in `../../HANDOFF.md`
(2026-08-27). That entry has **no evidenced cause** — an earlier version tied
it to the 2026-08-24 long-clip failure and to host RAM, and both were wrong
(the long-clip failure was a client-side idle TCP reap and is resolved). This
script exists to ANSWER the question, not to confirm a theory: the cgroup
`oom_kill` counter settles the RAM question in either direction.

It only blocks tools *outside* the container. The numbers that settle it are
readable from `/proc` and `/sys/fs/cgroup` with no privilege, so this script
reads them from inside and prints them where `vastai logs` will show them.

**Nothing to set up** -- it's `COPY`'d into the image and called by
`vast-entrypoint.sh` before the model fetch. It needs a rebuild to appear on a
worker; on an image without it the entrypoint prints one line and carries on.

**What it gives you, in the boot banner:**

| Line | Read it as |
|---|---|
| `BOOT #1` on an instance id you've seen before | container was **RECREATED** -- fresh writable layer. Explains a full ~40GB model re-fetch, and means `.has_benchmark` is gone. |
| `BOOT #2`, `#3`, … | container was **RESTARTED** -- `/workspace` survived, so `backend.__run_benchmark` will skip the benchmark and reuse the **cached** `max_perf`. If workload units changed since, this worker is reporting perf in the old units. |
| `cgroup oom_kill counter: <n>` non-zero | the kernel OOM-killed something in this cgroup. Host-RAM exhaustion, confirmed, in one number. |
| `[forensics][prev] …` replay lines | the last ~40 memory samples from the run that died -- cgroup usage vs limit, `MemAvailable`, and the top-3 RSS processes, right up to the kill. |

**Env knobs** (all optional, sane defaults):

| Var | Default | Meaning |
|---|---|---|
| `FORENSICS_DIR` | `/workspace/.boot-forensics` | State dir. Must be on a path that survives a container restart for the counter and replay to mean anything. |
| `FORENSICS_SAMPLE_INTERVAL` | `5` | Seconds between memory samples. |
| `FORENSICS_REPLAY_LINES` | `40` | How many of the previous run's samples to replay at boot. |

The sample log self-trims at 20k lines (~28h at 5s), so it can't grow into the
disk ComfyUI needs.

**Why it's called as a subprocess and not `source`d:** `vast-entrypoint.sh`
runs under `set -euo pipefail`. `errexit` isn't inherited across fork+exec, so
a subprocess call (each guarded with `|| true`) means nothing in a diagnostic
can ever be the reason a worker fails to boot. The sampler still outlives it --
it's backgrounded, orphaned when the script exits, and reparented to whatever
becomes PID 1 after `exec /start_server.sh`.

## Design decisions and their tradeoffs (also documented inline in worker.py)

**Healthcheck is `GET /`, not a real ComfyUI check.** RunPod's local
hosted-API server (`runpod.serverless.modules.rp_fastapi`) exposes exactly
four routes -- `/run`, `/runsync`, `/stream/{id}`, `/status/{id}`, all
POST -- plus a `GET /` that serves its Swagger docs page. There's no
`/health`. We use `GET /` as the PyWorker healthcheck because it's cheap,
side-effect-free, and always 200 while the FastAPI process is alive -- but
it only proves handler.py's HTTP layer is up, not that ComfyUI underneath
it is still healthy. A ComfyUI crash *after* a successful boot (e.g.
mid-generation CUDA OOM) won't flip the healthcheck; it'll only show up as
the next real request failing. This was the one genuinely open question
from the earlier research pass, and it resolves cleanly: `vastai`'s
`WorkerConfig`/`HandlerConfig`/healthcheck are all relative paths hit
against a *single* `model_server_url:model_server_port` (confirmed by
reading `vastai`'s own `backend.py` -- both the proxied route and the
healthcheck reuse the same `aiohttp.ClientSession` with that as its
`base_url`), so a healthcheck against ComfyUI's own port (separate from
handler.py's port) was never actually possible with this SDK -- not a
workaround to build, just not on the table.

**The startup benchmark runs the real pipeline once (or twice).** Vast
requires exactly one `HandlerConfig` with a `BenchmarkConfig`, and a
worker isn't marked ready until that benchmark succeeds -- there's no fast
synthetic workflow that exercises the same InfiniteTalk/CosyVoice nodes,
and a benchmark that doesn't actually run the pipeline wouldn't have
caught any of the real failures logged in HANDOFF.md. `do_warmup=True`
(current default, matching Vast's own Wan 2.2 example) means one warmup
generation (real model load into VRAM, slow) plus one measured generation
-- two full video generations before the worker is marked ready. Flip
`do_warmup` to `False` in `worker.py` to halve cold-start time if the
measured throughput number isn't worth it for this pipeline's request
volume.

**`max_queue_time=3600.0`, not the Wan example's `10.0`.** This governs
admission (PyWorker rejects new requests with HTTP 429 once *estimated*
wait time -- queued workload / measured throughput -- exceeds it), not a
per-request timer. Wan's image-scale example expects requests measured in
seconds; ours takes minutes, so 10s would reject nearly everything queued
behind an in-flight job.

**`allow_parallel_requests=False` forces benchmark concurrency to 1
regardless of `BenchmarkConfig.concurrency`** (confirmed in `vastai`'s
`backend.__run_benchmark`) -- set explicitly in `worker.py` anyway so the
intent survives an SDK version bump, not because it changes behavior today.

## What's confirmed vs. what to verify on first real deploy

Confirmed by reading `vastai` 1.5.4's actual source (downloaded from PyPI,
not assumed from docs) and by running `handler.py --rp_serve_api` locally:
the `WorkerConfig`/`HandlerConfig`/`BenchmarkConfig`/`LogActionConfig`
field names and semantics used in `worker.py`, the single-base-URL
healthcheck constraint, the benchmark concurrency-forcing behavior, the
`max_queue_time` admission semantics, and handler.py's actual HTTP
behavior (`GET /` -> 200, `POST /runsync` -> 200-even-on-error). Also
confirmed, 2026-08-15, once this folder moved into `vast/serverless/` and
the real `reassemble-models.sh` was reachable: it ends with `exec "$@"`.

Not verifiable from this sandbox, worth checking on the first real Vast
deploy: whether `CONTAINER_ID` is really always pre-set on Vast
Serverless instances; and general end-to-end timing (cold start with two
full video generations back-to-back could plausibly hit Vast's
platform-level boot-timeout, if one exists and is short -- not something
visible from outside a real endpoint).
