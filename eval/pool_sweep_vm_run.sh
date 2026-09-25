#!/usr/bin/env bash
# pool_sweep_vm_run.sh — run mu_eval/pool_window_sweep.py on mu-dev-vm (CLAUDE.md rule 13).
# Isolated remote dir (mu-core-poolsweep) so this lane's VM sync never collides with a concurrent
# lane's own vm_run.sh sync (root CLAUDE.md rule: worktree isolation). Same locking/rsync/reclaim
# discipline as eval/mem0_h2h/vm_run.sh, copied deliberately.
set -euo pipefail

_LOCK="/tmp/.mu_vm_test.lock"
exec 9>"$_LOCK"
flock -s -w "${MU_VM_LOCK_WAIT_S:-2700}" 9 || { echo "pool_sweep_vm_run.sh: VM lock timeout" >&2; exit 75; }

ZONE=europe-west3-b
PROJECT=amplified-vim-504612-s4
VM=mu-dev-vm
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_KEY="$HOME/.ssh/google_compute_engine"
SCRIPT="$1"; shift
MU_EVAL_DATA="${MU_EVAL_DATA:-/home/user/D/abstract_project/mma/data/locomo/locomo10.json}"
REMOTE_DIR="${POOLSWEEP_REMOTE_DIR:-mu_project/mu-core-poolsweep}"

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

"${SSH[@]}" "touch \$HOME/.mu_reclaim_hold" || true
release_hold() {
  ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
    "unlink \$HOME/.mu_reclaim_hold" >/dev/null 2>&1 || true
}
trap release_hold EXIT

FORWARD=""
for name in POOLSWEEP_OUT POOLSWEEP_SAMPLE POOLSWEEP_DATASET POOLSWEEP_IMPORTANCE \
            POOLSWEEP_POOLS_L10 POOLSWEEP_POOLS_L20 POOLSWEEP_REPEATS ${MU_FORWARD:-}; do
  value="${!name:-}"
  [[ -n "$value" ]] && FORWARD+="export $name=$(printf '%q' "$value"); "
done

echo "[3/3] running: $SCRIPT"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; \
  export POOLSWEEP_EVAL_DIR=\$HOME/$REMOTE_DIR/eval; ${FORWARD} \
  cd \$HOME/$REMOTE_DIR && PYTHONPATH=eval uv run python $SCRIPT $*"

echo "[done] fetching result…"
if [[ -n "${POOLSWEEP_OUT:-}" ]]; then
  mkdir -p "$(dirname "$POOLSWEEP_OUT")"
  scp -q -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP:$POOLSWEEP_OUT" "$POOLSWEEP_OUT" || \
    echo "!! could not scp $POOLSWEEP_OUT back — fetch manually"
fi
