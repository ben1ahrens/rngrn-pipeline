#!/usr/bin/env bash
# Driver: real-stripes frames x 1 seed, serial CUDA, 4-way parallel over frames.
# Run THROUGH scripts/guarded_run.sh (one flock hold for the whole batch):
#   bash scripts/guarded_run.sh bash experiments/real_stripes/run_frames.sh
# Torch threads pinned to 1 per process: CPU is in use elsewhere (owner, 2026-08-19);
# the serial-CUDA path needs the CPU only for python-side stepping.
#
# HISTORY OF THE 2026-08-19 BATCH (why this file looks the way it does):
#   Launch 1 (frames 1-4, timeout 3600): frames 1,3,4 killed at the wire (exit 124)
#     — under 4-way GPU contention a frame takes ~57-84 min. Frame 2 finished.
#   Launch 2 (frames 1,3,4,5,6,7,8, timeout 7200): frames 1,3,4,5 all started within
#     the SAME wall-clock second, and run dirs are named
#     <run_name>_<UTC-second>_seedN — so all four wrote to ONE directory
#     (stripes_partial_20260819_183655_seed0) and the last finisher's checkpoint
#     (frame 1's) clobbered the other three. Frames 3,4,5 were NOT rerun (owner
#     decision, 2026-08-19: not enough time), so the final analysis covers frames
#     1,2,6,7,8 only. Their per-frame rows in target_reports.jsonl remain valid;
#     only the on-disk checkpoints were lost.
#   The fix below — a per-sample tracking.run_name override — makes the run-dir
#   name unique per frame regardless of start second. Any future rerun MUST keep it.
set -u
cd "$(dirname "$0")/../.."
export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
  s=$1
  echo "FRAME $s START $(date -Is)"
  # 7200s: see history above — 3600s killed 3 of 4 frames at the wire.
  timeout 7200 .venv/bin/python -m rngrn.cli \
    --runs-root experiments/real_stripes target-report \
    --config configs/nc1_stripes_partial.yaml \
    --dataset-id stripes_colony_2ch --sample-key "$s" \
    --form nc1 --seeds 0 --workers 1 \
    -o tracking.run_name="stripes_${s}" \
    > "experiments/real_stripes/${s}.log" 2>&1
  echo "FRAME $s EXIT $? $(date -Is)"
}
export -f run_one

# As-run frame list of launch 2 was: 1 3 4 5 6 7 8 (frame 2 done in launch 1).
# Edit this list for any rerun; the run_name override above prevents collisions.
printf 'sample_%04d\n' 1 3 4 5 6 7 8 | xargs -P4 -I{} bash -c 'run_one {}'
echo "ALL FRAMES DONE $(date -Is)"
