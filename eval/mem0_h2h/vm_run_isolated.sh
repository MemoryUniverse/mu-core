#!/usr/bin/env bash
# vm_run_isolated.sh — run ONE eval script on mu-dev-vm from a worktree, in an ISOLATED sibling pair.
#
# WHY THIS EXISTS (AD-313). `vm_run.sh` syncs the worktree to `$HOME/mu_project/<dir>` and runs
# `uv sync` there. That worked for AD-309 and STOPS WORKING the moment any lane has also synced
# `mu-sdk-python` to the VM — which several now have. The reason is a hard-coded relative path,
# not a transient:
#
#   mu-core/pyproject.toml:28   mu-sdk       = { path = "../mu-sdk-python", editable = true }
#   mu-sdk-python/pyproject.toml:86  mu-contracts = { path = "../mu-core/packages/mu-contracts", ... }
#
# So a mu-core checkout at `mu_project/mu-core-h2h` resolves `mu-contracts` TWICE, from two
# different paths — `mu-core-h2h/packages/mu-contracts` (workspace member) and
# `mu-core/packages/mu-contracts` (via the SDK's own source) — and uv refuses:
#   "Requirements contain conflicting URLs for package `mu-contracts`".
# MEASURED 2026-09-26: `vm_run.sh` fails at step 2/3 with exactly that error, for BOTH the default
# remote dir and a fresh one, on a VM where `mu_project/mu-sdk-python` is present. It is not a
# property of the worktree; it is a property of the directory NAME.
#
# The fix is to give the run its own parent directory in which the mu-core checkout IS named
# `mu-core` and the SDK sits beside it, so `../mu-sdk-python` and `../mu-core/...` both resolve
# back into this run's own pair. No shared directory is written and nothing another lane is using
# is touched — which is the other half of the requirement, since `vm_run.sh`'s `rsync --delete`
# into a shared path is itself a way to corrupt a sibling lane's tree mid-run.
#
#   H2H_OUT=/tmp/x.json ./eval/mem0_h2h/vm_run_isolated.sh eval/mem0_h2h/verify_efficiency.py --out /tmp/x.json
#
# Everything else is deliberately identical to `vm_run.sh`: the same shared `flock`, the same rsync
# exclusions, the same `~/.mu_reclaim_hold` released in a trap, the same explicit env forwarding
# (an eval run must never inherit whatever the invoking shell happened to have set).
set -euo pipefail

_LOCK="/tmp/.mu_vm_test.lock"
exec 9>"$_LOCK"
flock -s -w "${MU_VM_LOCK_WAIT_S:-2700}" 9 || { echo "vm_run_isolated.sh: VM lock timeout" >&2; exit 75; }

ZONE=europe-west3-b
PROJECT=amplified-vim-504612-s4
VM=mu-dev-vm
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Walk UP from this checkout to find the `mu_project` root that holds `mu-sdk-python`. Not a fixed
# `../..`: this script is run both from the main `mu-core` checkout (one level down from the root)
# and from a worktree under `mu-core/.worktrees/<name>` (three levels down), and a hard-coded depth
# silently resolves to a directory that does not exist -- measured, that is the first thing it did.
_find_sdk() {
  local d="$1"
  while [[ "$d" != "/" ]]; do
    [[ -d "$d/mu-sdk-python" ]] && { printf '%s\n' "$d/mu-sdk-python"; return 0; }
    d="$(dirname "$d")"
  done
  return 1
}
SIBLING_SDK="${MU_SDK_PY:-$(_find_sdk "$REPO_ROOT" || true)}"
SSH_KEY="$HOME/.ssh/google_compute_engine"
SCRIPT="$1"; shift
MU_EVAL_DATA="${MU_EVAL_DATA:-/home/user/D/abstract_project/mma/data/locomo/locomo10.json}"
# The isolated PARENT; the checkout inside it is always named `mu-core` (that is the whole point).
LANE="${H2H_LANE:-ad313}"
PARENT="lanes/$LANE"
REMOTE_DIR="$PARENT/mu-core"

[[ -d "$SIBLING_SDK" ]] || { echo "!! mu-sdk-python not found at $SIBLING_SDK (set MU_SDK_PY)"; exit 1; }

IP=$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
[[ -n "$IP" ]] || { echo "!! VM has no external IP — run infra/mu-vm/vm_reup.sh"; exit 1; }
SSH=(ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP")
RSYNC_E=(-e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no")
EXCL=(--exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache'
      --exclude '.mypy_cache' --exclude '.trash' --exclude '.claude' --exclude '.mu_data'
      --exclude 'logs' --exclude 'graphify-out' --exclude '.git' --exclude '.worktrees')

echo "[1/3] syncing the isolated pair -> VM:$PARENT/{mu-core,mu-sdk-python} …"
"${SSH[@]}" "mkdir -p \$HOME/$PARENT \$HOME/mu_eval_data"
rsync -az --delete "${EXCL[@]}" "${RSYNC_E[@]}" "$REPO_ROOT/" "user@$IP:$REMOTE_DIR/"
rsync -az --delete "${EXCL[@]}" "${RSYNC_E[@]}" "$SIBLING_SDK/" "user@$IP:$PARENT/mu-sdk-python/"
rsync -az "${RSYNC_E[@]}" "$MU_EVAL_DATA" "user@$IP:mu_eval_data/"

echo "[2/3] uv sync…"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; cd \$HOME/$REMOTE_DIR && uv sync >/dev/null 2>&1 || uv sync"

"${SSH[@]}" "touch \$HOME/.mu_reclaim_hold" || true
release_hold() {
  ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
    "unlink \$HOME/.mu_reclaim_hold" >/dev/null 2>&1 || true
}
trap release_hold EXIT

FORWARD=""
for name in H2H_OUT H2H_LABEL H2H_LIMITS H2H_SAMPLE H2H_DATASET H2H_IMPORTANCE H2H_REPEATS \
            H2H_WARMUP ${MU_FORWARD:-}; do
  value="${!name:-}"
  [[ -n "$value" ]] && FORWARD+="export $name=$(printf '%q' "$value"); "
done

# Re-quote every argument with `printf %q` before it is spliced into the remote shell string.
# `$*` alone loses quoting: `-m "not integration"` arrives as three words and pytest reads
# `integration` as a path, reporting "file or directory not found: integration" and running ZERO
# tests -- a green-looking line for a suite that never ran. Measured; `vm_run.sh` has the same
# latent bug for any argument containing a space.
ARGV=""
for a in "$@"; do ARGV+=" $(printf '%q' "$a")"; done

echo "[3/3] running: $SCRIPT$ARGV"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; \
  export H2H_EVAL_DIR=\$HOME/$REMOTE_DIR/eval; ${FORWARD} \
  cd \$HOME/$REMOTE_DIR && PYTHONPATH=eval uv run python $(printf '%q' "$SCRIPT")$ARGV"
