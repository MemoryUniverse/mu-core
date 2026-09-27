#!/usr/bin/env bash
# vm_run.sh — run ONE arbitrary eval script on mu-dev-vm (CLAUDE.md rule 13: never on the laptop).
#
# `eval/vm_eval.sh` only knows how to run `python -m mu_eval`; a one-off measurement script is not
# a `mu_eval` subcommand. Everything else about the discipline is copied from it deliberately:
# the SAME lock (/tmp/.mu_vm_test.lock), now taken EXCLUSIVELY (AD-336), the SAME rsync exclusions, the SAME
# `~/.mu_reclaim_hold` so the VM's 20-minute collection-reclaim cron cannot delete the Qdrant
# collections out from under a live run (that really happened — ARCHITECTURE-DELTAS AD-201).
#
# KNOWN LIMIT: `.git` is excluded. This runner is pointed at a git WORKTREE, whose `.git` is a
# pointer FILE naming a gitdir under the laptop's main checkout — a path that does not exist on the
# VM, so shipping it makes every `git` call there fail with "not a git repository" rather than
# simply being absent. `eval/tests/test_provenance_unit.py::test_git_revision_reports_the_real_head
# _of_this_checkout` asserts against a REAL checkout by construction and therefore cannot run
# through this script; it runs through `infra/mu-vm/vm_test.sh mu-core`, which syncs the main
# checkout, and that is the path CI uses.
#
#   H2H_OUT=/tmp/x.json H2H_LABEL=A ./eval/mem0_h2h/vm_run.sh eval/mem0_h2h/ours_arm.py
set -euo pipefail

_LOCK="/tmp/.mu_vm_test.lock"
exec 9>"$_LOCK"
# AD-336: **EXCLUSIVE, not shared.** This was `flock -s` and that is why two eval arms have been
# observed running at once on this one VM — AD-333 recorded exactly that ("AD-332's base arm ran
# 15:15:14 -> 15:17:46 — *inside* the LLM arm's 15:12:31 -> 15:26:19 window") and diagnosed it as a
# hold-file OWNERSHIP problem. Ownership was half of it. The reason two arms overlapped at all is
# this lock: `-s` is a READ lock, so every caller got it immediately and nothing ever queued.
# Concurrency is not merely untidy here, it invalidates the measurement three ways: each wrapper
# `rsync --delete`s into ONE shared `$REMOTE_DIR` and runs `uv sync` in its `.venv` (so one arm
# rewrites the tree another is executing from), both ingest into Qdrant/FalkorDB on the same box,
# and every p50/p95 this harness reports is measured against whatever else was running. An eval arm
# is a whole-box workload, exactly like `vm_test.sh`'s full-suite case, and takes the same `-x`.
# A second caller WAITS (bounded by MU_VM_LOCK_WAIT_S) instead of piling on.
flock -x -w "${MU_VM_LOCK_WAIT_S:-2700}" 9 || { echo "vm_run.sh: VM lock timeout (another eval arm or suite holds the VM)" >&2; exit 75; }

ZONE=europe-west3-b
PROJECT=amplified-vim-504612-s4
VM=mu-dev-vm
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_KEY="$HOME/.ssh/google_compute_engine"
SCRIPT="$1"; shift
MU_EVAL_DATA="${MU_EVAL_DATA:-/home/user/D/abstract_project/mma/data/locomo/locomo10.json}"
REMOTE_DIR="${H2H_REMOTE_DIR:-mu_project/mu-core-h2h}"

IP=$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
[[ -n "$IP" ]] || { echo "!! VM has no external IP — run infra/mu-vm/vm_reup.sh"; exit 1; }
SSH=(ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP")

echo "[1/3] syncing worktree -> VM:$REMOTE_DIR …"
"${SSH[@]}" "mkdir -p \$HOME/$REMOTE_DIR \$HOME/mu_eval_data"
rsync -az --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '.mypy_cache' --exclude '.trash' --exclude '.claude' --exclude '.mu_data' \
  --exclude 'logs' --exclude 'graphify-out' --exclude '.git' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$REPO_ROOT/" "user@$IP:$REMOTE_DIR/"
rsync -az -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$MU_EVAL_DATA" "user@$IP:mu_eval_data/"

echo "[2/3] uv sync…"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; cd \$HOME/$REMOTE_DIR && uv sync >/dev/null 2>&1 || uv sync"

# The VM's own reclaim cron deletes every mu_mtm__*/mu_g__* every 20 minutes; the hold stops it
# eating this run's collections mid-flight. Released in a trap so a crash cannot leave the sweep
# disabled (the reclaim also ignores a hold older than MU_RECLAIM_HOLD_MAX_S).
#
# A PER-RUN entry under `~/.mu_reclaim_hold.d/`, not the shared single file (AD-333): three
# wrappers touched and unlinked the same `~/.mu_reclaim_hold`, so one finishing released the
# protection another still needed — and that is how AD-332's 6/150 got measured against a
# collection the sweep had deleted 2 minutes earlier.
# Named from THIS shell's `$$` and then reused verbatim on both sides — `\$\$` in the remote
# string would be the REMOTE shell's pid and would never match what the trap removes, leaving a
# hold entry behind on every run (which the 4 h cap would eventually expire, but only after
# disabling the sweep for 4 h).
HOLD_ENTRY=".mu_reclaim_hold.d/vm_run.$$"
"${SSH[@]}" "mkdir -p \$HOME/.mu_reclaim_hold.d && touch \$HOME/$HOLD_ENTRY" || true
release_hold() {
  ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
    "rm -f \$HOME/$HOLD_ENTRY" >/dev/null 2>&1 || true
}
trap release_hold EXIT

# Only names this script is told to forward reach the VM — an eval run must never silently
# inherit whatever the invoking shell happened to have set (vm_eval.sh's own rule).
FORWARD=""
# H2H_OUT is deliberately NOT in this list — it is rewritten to a VM-side path below. (It is
# excluded here rather than substituted out afterwards: `${FORWARD//export H2H_OUT=*; /}` looks
# like the obvious way to do that and is wrong, because bash glob substitution is greedy and `*; `
# swallows every later export in the string.)
for name in H2H_LABEL H2H_LIMITS H2H_SAMPLE H2H_DATASET H2H_IMPORTANCE ${MU_FORWARD:-}; do
  value="${!name:-}"
  [[ -n "$value" ]] && FORWARD+="export $name=$(printf '%q' "$value"); "
done

# AD-336: `H2H_OUT` is opened by the script ON THE VM, so a caller's LAPTOP path — which is what
# every driver in this repo passes, because that is where the artifact belongs — is a directory the
# VM does not have. `ours_arm.py` writes its export as the LAST thing it does, so the arm ran
# correctly, printed its `gold_in_context`, and THEN died with `FileNotFoundError`, exit 1, no
# artifact. Two arms of this pass lost their per-row export that way, and the driver's
# "non-zero rc means the harness refused this measurement" rule flagged a perfectly good arm.
#
# So the remote run writes to a VM-side temp path and this script copies the file back to where the
# caller asked for it. The caller's own path is never handed to the remote shell.
REMOTE_OUT=""
if [[ -n "${H2H_OUT:-}" ]]; then
  REMOTE_OUT="/tmp/h2h_out.$$.$(basename "$H2H_OUT")"
  FORWARD+="export H2H_OUT=$(printf '%q' "$REMOTE_OUT"); "
  mkdir -p "$(dirname "$H2H_OUT")"
fi
fetch_out() {
  [[ -n "$REMOTE_OUT" ]] || return 0
  if scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP:$REMOTE_OUT" "$H2H_OUT" 2>/dev/null
  then echo "[out] $H2H_OUT"; else echo "[out] !! no export at $REMOTE_OUT on the VM" >&2; fi
  ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
    "rm -f $(printf '%q' "$REMOTE_OUT")" >/dev/null 2>&1 || true
}
trap 'fetch_out; release_hold' EXIT

echo "[3/3] running: $SCRIPT"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; \
  export H2H_EVAL_DIR=\$HOME/$REMOTE_DIR/eval; ${FORWARD} \
  cd \$HOME/$REMOTE_DIR && PYTHONPATH=eval uv run python $SCRIPT $*"
