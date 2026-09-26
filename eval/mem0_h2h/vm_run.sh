#!/usr/bin/env bash
# vm_run.sh — run ONE arbitrary eval script on mu-dev-vm (CLAUDE.md rule 13: never on the laptop).
#
# `eval/vm_eval.sh` only knows how to run `python -m mu_eval`; a one-off measurement script is not
# a `mu_eval` subcommand. Everything else about the discipline is copied from it deliberately:
# the SAME shared lock (/tmp/.mu_vm_test.lock), the SAME rsync exclusions, the SAME
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
flock -s -w "${MU_VM_LOCK_WAIT_S:-2700}" 9 || { echo "vm_run.sh: VM lock timeout" >&2; exit 75; }

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

# The VM's own reclaim cron deletes every mu_mtm__*/mu_g__* every 20 minutes; the hold file stops
# it eating this run's collections mid-flight. Released in a trap so a crash cannot leave the
# sweep disabled (the reclaim also ignores a hold older than MU_RECLAIM_HOLD_MAX_S).
"${SSH[@]}" "touch \$HOME/.mu_reclaim_hold" || true
release_hold() {
  ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
    "unlink \$HOME/.mu_reclaim_hold" >/dev/null 2>&1 || true
}
trap release_hold EXIT

# Only names this script is told to forward reach the VM — an eval run must never silently
# inherit whatever the invoking shell happened to have set (vm_eval.sh's own rule).
FORWARD=""
for name in H2H_OUT H2H_LABEL H2H_LIMITS H2H_SAMPLE H2H_DATASET H2H_IMPORTANCE ${MU_FORWARD:-}; do
  value="${!name:-}"
  [[ -n "$value" ]] && FORWARD+="export $name=$(printf '%q' "$value"); "
done

echo "[3/3] running: $SCRIPT"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; \
  export H2H_EVAL_DIR=\$HOME/$REMOTE_DIR/eval; ${FORWARD} \
  cd \$HOME/$REMOTE_DIR && PYTHONPATH=eval uv run python $SCRIPT $*"
