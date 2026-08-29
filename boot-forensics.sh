#!/bin/bash
# Boot forensics for the Vast Serverless worker -- makes the "instance
# silently reboots" failure observable without SSH.
#
# WHY THIS EXISTS (see HANDOFF.md, 2026-08-27 unexplained container restart --
# cause UNKNOWN; an earlier version of this header tied it to the long-clip
# failure and to host RAM, and both were wrong -- the long-clip failure was a
# client-side idle TCP reap and is resolved):
# a worker dies with NO Python output at all, the container comes back, the
# whole entrypoint re-runs, and `vastai logs` has already been reset by the
# reboot -- so the evidence for what killed it is destroyed by the thing that
# killed it. HANDOFF records this as unobservable because serverless workers
# can't be SSH'd into: no dmesg, no free, no nvidia-smi at the moment of death.
#
# That is only true for tools OUTSIDE the container. We own everything inside
# it. This script does three things, all from inside:
#
#   1. BOOT COUNTER, on a path that survives a container restart. Answers the
#      question HANDOFF currently has to guess at: was that a container
#      RESTART (writable layer intact, counter reads 2, 3, ...) or a fresh
#      container RECREATE (counter reads 1 every time)? That single number
#      also settles whether a stale `.has_benchmark` in /workspace/vast-pyworker
#      can be serving old perf units after a restart -- see the scaling
#      section in HANDOFF.md.
#
#   2. REPLAY OF THE PREVIOUS RUN'S LAST MOMENTS. The memory log is rotated at
#      boot and its tail printed to stdout, so `vastai logs` AFTER the reboot
#      shows the RAM curve leading right up to the kill -- exactly the window
#      the reboot used to erase.
#
#   3. A BACKGROUND MEMORY SAMPLER. Every few seconds: cgroup memory usage vs
#      the container's limit, MemAvailable, swap, the cgroup OOM-kill counter,
#      and the top RSS processes. If the leading hypothesis is right (host RAM
#      exhaustion -> kernel SIGKILL -> container restart), this shows the
#      approach to the ceiling and names the process holding it.
#
# Deliberately best-effort and non-fatal: vast-entrypoint.sh runs under
# `set -euo pipefail`, so every call site must be guarded with `|| true` and
# nothing in here may exit non-zero on a box where a path does not exist.
# Diagnostics must never be the reason a worker fails to boot.

FORENSICS_DIR="${FORENSICS_DIR:-/workspace/.boot-forensics}"
SAMPLE_INTERVAL="${FORENSICS_SAMPLE_INTERVAL:-5}"
REPLAY_LINES="${FORENSICS_REPLAY_LINES:-40}"

_f_log() { echo "[forensics] $*"; }

# --- cgroup v2 / v1 abstraction ---------------------------------------------
# Vast hosts have been seen on both, and at least one box exposes a v2
# memory.current alongside a v1 memory/ tree. Read whichever exists; print "?"
# rather than failing if neither does.
_f_cg_read() {
    # $1 = cgroup v2 filename, $2 = cgroup v1 relative path
    if [ -r "/sys/fs/cgroup/$1" ]; then
        cat "/sys/fs/cgroup/$1" 2>/dev/null || echo "?"
    elif [ -r "/sys/fs/cgroup/memory/$2" ]; then
        cat "/sys/fs/cgroup/memory/$2" 2>/dev/null || echo "?"
    else
        echo "?"
    fi
}

_f_mem_current() { _f_cg_read "memory.current" "memory.usage_in_bytes" | head -n1; }
_f_mem_max()     { _f_cg_read "memory.max"     "memory.limit_in_bytes" | head -n1; }

# OOM-kill counter. cgroup v2 exposes it in memory.events; v1 has no direct
# equivalent, so fall back to memory.failcnt (allocation-failure count -- not
# the same thing, but it moves for the same reason and beats nothing).
_f_oom_count() {
    if [ -r /sys/fs/cgroup/memory.events ]; then
        awk '/^oom_kill /{print $2; found=1} END{if(!found) print "0"}' \
            /sys/fs/cgroup/memory.events 2>/dev/null || echo "?"
    elif [ -r /sys/fs/cgroup/memory/memory.failcnt ]; then
        echo "failcnt:$(cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null || echo '?')"
    else
        echo "?"
    fi
}

_f_meminfo() {
    awk '/^MemAvailable:/{a=$2} /^MemTotal:/{t=$2} /^SwapFree:/{sf=$2} /^SwapTotal:/{st=$2}
         END{printf "memavail_kb=%s memtotal_kb=%s swapfree_kb=%s swaptotal_kb=%s", a, t, sf, st}' \
        /proc/meminfo 2>/dev/null || echo "memavail_kb=? memtotal_kb=? swapfree_kb=? swaptotal_kb=?"
}

# Top 3 processes by RSS, as "name:rss_kb" -- names the actual holder when the
# ceiling is hit (expect python3/ComfyUI; anything else is itself a finding).
_f_top_rss() {
    ps -eo rss=,comm= 2>/dev/null \
        | sort -rn \
        | head -n3 \
        | awk '{printf "%s:%s ", $2, $1}' \
        || true
}

# --- 1 + 2: boot banner, counter, and previous-run replay --------------------
forensics_boot() {
    mkdir -p "$FORENSICS_DIR" 2>/dev/null || {
        _f_log "WARNING: cannot create $FORENSICS_DIR -- forensics disabled this boot"
        return 0
    }

    local counter_file="$FORENSICS_DIR/boot_count"
    local mem_log="$FORENSICS_DIR/mem.log"
    local prev_log="$FORENSICS_DIR/mem.prev.log"

    # NB: written as an `if`, not `[ -r f ] && count=...`. That form returns
    # non-zero when the file is absent, which under an inherited `set -e`
    # would abort this function on the very first boot -- silently disabling
    # the diagnostic exactly when it is most needed.
    local count=0
    if [ -r "$counter_file" ]; then
        count=$(cat "$counter_file" 2>/dev/null || echo 0)
    fi
    case "$count" in (*[!0-9]*|"") count=0 ;; esac
    count=$((count + 1))
    echo "$count" > "$counter_file" 2>/dev/null || true

    _f_log "=============================================================="
    _f_log "BOOT #$count in this container's writable layer"
    if [ "$count" -eq 1 ]; then
        _f_log "  -> counter absent/reset. Either a genuinely new instance, or"
        _f_log "     the container was RECREATED (fresh writable layer), not"
        _f_log "     restarted. If the instance id is unchanged from an earlier"
        _f_log "     boot, that is the RECREATE case -- and /workspace state"
        _f_log "     (venv, cloned pyworker, .has_benchmark, fetched models) is"
        _f_log "     gone, which is what a full model re-fetch means."
    else
        _f_log "  -> the writable layer SURVIVED. This was a container RESTART."
        _f_log "     /workspace state persisted, so .has_benchmark is still"
        _f_log "     there and backend.__run_benchmark will SKIP the benchmark"
        _f_log "     and reuse the cached max_perf. If workload units changed"
        _f_log "     since that file was written, this worker is now reporting"
        _f_log "     perf in the OLD units. See HANDOFF.md, scaling section."
    fi
    _f_log "  cgroup mem: current=$(_f_mem_current) max=$(_f_mem_max)"
    _f_log "  cgroup oom_kill counter: $(_f_oom_count)   <-- non-zero means the"
    _f_log "     kernel OOM-killed something in this cgroup. That is the answer."
    _f_log "  $(_f_meminfo)"
    _f_log "=============================================================="

    # Replay: rotate last run's samples and print the tail, so the reboot no
    # longer erases the run-up to the kill.
    if [ -s "$mem_log" ]; then
        mv -f "$mem_log" "$prev_log" 2>/dev/null || true
    fi
    if [ -s "$prev_log" ]; then
        _f_log "---- last $REPLAY_LINES memory samples from the PREVIOUS run ----"
        tail -n "$REPLAY_LINES" "$prev_log" 2>/dev/null | while IFS= read -r line; do
            echo "[forensics][prev] $line"
        done
        _f_log "---- end of previous run ----"
    else
        _f_log "no previous-run memory samples on disk (first boot, or recreate)"
    fi
}

# --- 3: background sampler ---------------------------------------------------
forensics_start_sampler() {
    local mem_log="$FORENSICS_DIR/mem.log"
    mkdir -p "$FORENSICS_DIR" 2>/dev/null || return 0

    (
        while true; do
            printf '%s cur=%s max=%s oom=%s %s top=[%s]\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "$(_f_mem_current)" \
                "$(_f_mem_max)" \
                "$(_f_oom_count)" \
                "$(_f_meminfo)" \
                "$(_f_top_rss)" \
                >> "$mem_log" 2>/dev/null || true
            # Keep the file bounded -- a multi-hour batch at 5s is ~2MB, but a
            # worker that lives for days must not fill the disk ComfyUI needs.
            # Trim to the most recent 20k samples (~28h at 5s).
            if [ "$(wc -l < "$mem_log" 2>/dev/null || echo 0)" -gt 20000 ]; then
                tail -n 10000 "$mem_log" > "$mem_log.tmp" 2>/dev/null \
                    && mv -f "$mem_log.tmp" "$mem_log" 2>/dev/null
            fi
            sleep "$SAMPLE_INTERVAL"
        done
    ) &
    _f_log "memory sampler started (pid $!, every ${SAMPLE_INTERVAL}s -> $mem_log)"
}

# Allow both `source boot-forensics.sh` (for the functions) and direct
# execution (`boot-forensics.sh boot` / `... sampler` / no args = both).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-all}" in
        boot)    forensics_boot ;;
        sampler) forensics_start_sampler ;;
        *)       forensics_boot; forensics_start_sampler ;;
    esac
fi
