#!/usr/bin/env bash
# link_payloads.sh — provision a worktree's dataset payloads from the canonical store.
#
# WHY THIS EXISTS. `payload.h5` is gitignored and `manifest.json` is tracked, so a fresh
# worktree lists every dataset while holding none of the data. The obvious one-liner
#
#     for d in data/datasets/*/; do ln -sf $MAIN/$d/payload.h5 $d/payload.h5; done
#
# is WRONG in two ways, and both bit on 2026-07-29:
#   1. For a dataset generated inside a worktree and not yet harvested, the canonical
#      payload does not exist, so `ln -sf` happily creates a DANGLING symlink. Three of
#      four agents in a wave were then unable to load their data, and the failure surfaced
#      only as a FileNotFoundError deep inside h5py.
#   2. Re-running it CLOBBERS a real generated payload with a dangling link — destroying
#      data that existed nowhere else.
#
# So: link only when the canonical payload really exists, never overwrite a real file, and
# report anything missing loudly instead of leaving a broken link behind.
#
# Usage:  bash scripts/link_payloads.sh [canonical_datasets_root]
set -uo pipefail

MAIN="${1:-/home/benja/projects/personal/rngrn/rngrn-pipeline/data/datasets}"
here="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$here" || exit 1

if [ ! -d "$MAIN" ]; then
    echo "link_payloads: canonical store not found: $MAIN" >&2
    exit 1
fi
if [ "$(cd "$MAIN" && pwd -P)" = "$(cd data/datasets && pwd -P)" ]; then
    echo "link_payloads: this IS the canonical store; nothing to link."
    exit 0
fi

linked=0; kept=0; missing=()
for d in data/datasets/*/; do
    id="$(basename "$d")"
    tgt="$d/payload.h5"
    src="$MAIN/$id/payload.h5"

    # a real file here is either the canonical copy or freshly generated data — never touch it
    if [ -f "$tgt" ] && [ ! -L "$tgt" ]; then
        kept=$((kept + 1)); continue
    fi
    if [ ! -f "$src" ]; then
        rm -f "$tgt"                      # drop a dangling link rather than leave a trap
        missing+=("$id"); continue
    fi
    ln -sfn "$src" "$tgt" && linked=$((linked + 1))
done

echo "link_payloads: linked=$linked  kept_local=$kept  missing=${#missing[@]}"
if [ "${#missing[@]}" -gt 0 ]; then
    printf 'link_payloads: NO CANONICAL PAYLOAD for: %s\n' "${missing[*]}" >&2
    echo "  These datasets cannot be loaded. Either the payload was never harvested into" >&2
    echo "  $MAIN (see CLAUDE.md 6a), or it needs regenerating — docs/DATASETS_L.md records" >&2
    echo "  the generating command and seed for the L-decoupled sets." >&2
    exit 2
fi
