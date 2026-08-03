#!/usr/bin/env bash
# guarded_run.sh — run one trainer invocation under a memory guard.
#
# WHY THIS EXISTS. Five Claude Code sessions died between 2026-07-29 and 2026-08-03,
# each killing hours of GPU compute. The cause was diagnosed from /var/log/syslog: the
# WSL2 VM's Linux global OOM killer, five times, at
#
#   2026-07-29 23:21:14 | 2026-07-30 03:33:29 | 2026-08-03 15:37:44
#   2026-08-03 17:03:21 | 2026-08-03 18:47:57
#
# At EVERY event: free swap 0 kB and ~90 MiB free RAM. Python RSS was 13.2-14.3 GiB
# against a VM MemTotal of 15.34 GiB. Each trainer process is 1.47-1.68 GiB RSS, and
# 17-21 of them ran concurrently: two agents x (1 parent + --workers 4) x ~1.6 GiB is
# ~16 GiB, i.e. structurally above the ceiling before anything else loads.
#
# Two things this does NOT fix, recorded so nobody assumes otherwise:
#   * The VM ceiling itself. /mnt/c/Users/benja/.wslconfig has `processors=14` but NO
#     `memory=` key, so WSL2 silently defaults to 50 % of the 31.4 GiB host = 15.34 GiB.
#     Raising it is a Windows-side edit needing `wsl --shutdown`.
#   * The 20 -> 14 core reduction. It did not help and could not have: `processors=`
#     does not affect the memory ceiling, and the queue scripts pass --workers
#     explicitly, so fewer cores does not shrink the pool.
#
# WHAT IT DOES, in order:
#   1. Serialises trainer sweeps ACROSS WORKTREES via one flock. Per-agent limits cannot
#      help here -- the overcommit is the SUM over concurrently running agents, and no
#      agent can see the others.
#   2. Waits for MemAvailable to clear a floor before launching.
#   3. Raises its own oom_score_adj so that if the kernel does have to kill something it
#      picks THIS trainer rather than the session. Before this, trainers sat at adj 0
#      while session systemd/dbus-daemon sat at 100-200, so the kernel took down the
#      whole user session instead of one cell -- exactly backwards. Raising adj needs no
#      privilege (lowering would).
#
# Usage:  bash scripts/guarded_run.sh <command> [args...]
# Env:    RNGRN_MEM_FLOOR_MB   (default 4096) MemAvailable required before launching
#         RNGRN_MEM_WAIT_S     (default 1800) give up waiting after this long
#         RNGRN_LOCK_WAIT_S    (default 7200) give up waiting for the lock after this
#         RNGRN_LOCK_PATH      (default /home/benja/projects/personal/rngrn/.trainer.lock)
#         RNGRN_OOM_ADJ        (default 500) 0..1000; higher = killed first
#         RNGRN_GUARD_OFF=1    bypass entirely (say so in writing if you use it)
set -uo pipefail

if [[ $# -eq 0 ]]; then
    echo "guarded_run: no command given" >&2
    exit 2
fi

if [[ "${RNGRN_GUARD_OFF:-0}" == "1" ]]; then
    exec "$@"
fi

MEM_FLOOR_MB="${RNGRN_MEM_FLOOR_MB:-4096}"
MEM_WAIT_S="${RNGRN_MEM_WAIT_S:-1800}"
LOCK_WAIT_S="${RNGRN_LOCK_WAIT_S:-7200}"
LOCK_PATH="${RNGRN_LOCK_PATH:-/home/benja/projects/personal/rngrn/.trainer.lock}"
OOM_ADJ="${RNGRN_OOM_ADJ:-500}"

mem_available_mb() {
    awk '/^MemAvailable:/ {print int($2/1024); exit}' /proc/meminfo
}

# Raise our own OOM score so the kernel prefers us over the session. Children inherit it.
if ! echo "$OOM_ADJ" > /proc/self/oom_score_adj 2>/dev/null; then
    echo "guarded_run: WARNING could not set oom_score_adj (continuing unprotected)" >&2
fi

# 1. one trainer sweep at a time, across every worktree
exec 9>"$LOCK_PATH" || { echo "guarded_run: cannot open lock $LOCK_PATH" >&2; exit 3; }
if ! flock -w "$LOCK_WAIT_S" 9; then
    echo "guarded_run: FAILED to acquire $LOCK_PATH after ${LOCK_WAIT_S}s -- another sweep is still running" >&2
    exit 4
fi

# 2. wait for headroom. Deliberately AFTER the lock: holding the lock while waiting stops
#    a second sweep from starting into the same shortage.
waited=0
while (( $(mem_available_mb) < MEM_FLOOR_MB )); do
    if (( waited >= MEM_WAIT_S )); then
        echo "guarded_run: FAILED MemAvailable $(mem_available_mb)MB stayed below ${MEM_FLOOR_MB}MB for ${MEM_WAIT_S}s" >&2
        exit 5
    fi
    if (( waited % 60 == 0 )); then
        echo "guarded_run: waiting for memory -- MemAvailable $(mem_available_mb)MB < ${MEM_FLOOR_MB}MB (${waited}s)" >&2
    fi
    sleep 10
    waited=$(( waited + 10 ))
done

echo "guarded_run: START MemAvailable=$(mem_available_mb)MB oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null || echo '?') -- $*" >&2
"$@"
rc=$?
echo "guarded_run: END rc=$rc MemAvailable=$(mem_available_mb)MB" >&2
exit $rc
