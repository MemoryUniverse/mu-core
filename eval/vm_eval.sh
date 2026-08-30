#!/usr/bin/env bash
# vm_eval.sh — run the mu_eval harness ON THE VM (CLAUDE.md rule 13), never on the laptop.
#
#   ./eval/vm_eval.sh probe-scores --out /tmp/probe.json
#   ./eval/vm_eval.sh baseline --dataset "$DATA" --samples 2 --out /tmp/baseline.json
#
# WHY a second script instead of vm_test.sh: vm_test.sh runs *pytest*. The harness is a
# measurement CLI, not a test — it has no pass/fail, it produces numbers. Everything else about
# the discipline is identical and reused deliberately:
#
#   * the SAME lock file (/tmp/.mu_vm_test.lock) taken SHARED, so an eval run and a path-limited
#     suite can coexist but neither overlaps a full suite. The VM is one shared 32 GB box and
#     concurrent full suites have OOM-killed it at 27 GB RSS;
#   * the SAME rsync exclusions, so no .venv/.mu_data/logs cross the wire;
#   * `uv sync` before the run, so the far side is the code that was just shipped.
#
# The LoCoMo dataset is NOT in this repo (someone else's data, 2.8 MB, MIT). Point MU_EVAL_DATA at
# a local copy and this script ships it to ~/mu_eval_data/ on the VM once.
set -euo pipefail

_LOCK="/tmp/.mu_vm_test.lock"
exec 9>"$_LOCK"
if ! flock -s -w "${MU_VM_LOCK_WAIT_S:-2700}" 9; then
  echo "vm_eval.sh: waited >${MU_VM_LOCK_WAIT_S:-2700}s for the VM lock — refusing to pile on." >&2
  exit 75
fi
echo "[lock] VM lock acquired: shared (eval run, pid $$)"

ZONE=europe-west3-b
PROJECT=amplified-vim-504612-s4
VM=mu-dev-vm
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_KEY="$HOME/.ssh/google_compute_engine"
MU_EVAL_DATA="${MU_EVAL_DATA:-/home/user/D/abstract_project/mma/data/locomo/locomo10.json}"

IP=$(gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
[[ -n "$IP" ]] || { echo "!! VM has no external IP — is it stopped? try infra/mu-vm/vm_reup.sh"; exit 1; }
SSH=(ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP")

echo "[1/3] syncing mu-core -> VM…"
rsync -az --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '.mypy_cache' --exclude '.trash' --exclude '.claude' --exclude '.mu_data' \
  --exclude 'logs' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  "$REPO_ROOT/" "user@$IP:~/mu_project/mu-core/"

if [[ -f "$MU_EVAL_DATA" ]]; then
  echo "      + dataset $(basename "$MU_EVAL_DATA")"
  "${SSH[@]}" "mkdir -p ~/mu_eval_data"
  rsync -az -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "$MU_EVAL_DATA" "user@$IP:~/mu_eval_data/"
else
  echo "      ! MU_EVAL_DATA not found at $MU_EVAL_DATA — dataset-backed commands will fail loudly."
fi

echo "[2/3] uv sync…"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; cd ~/mu_project/mu-core && uv sync >/dev/null 2>&1 || uv sync"

# Engine knob overrides for a DIAGNOSTIC arm (e.g. MU_EVAL_ENV="MU_RECALL__FLOOR_PROTECT_LIMIT=0").
# Kept explicit rather than exporting the whole local environment: an eval run must not silently
# inherit whatever a shell happened to have set, or its numbers mean nothing.
REMOTE_ENV="${MU_EVAL_ENV:+export $MU_EVAL_ENV;}"
if [[ -n "$REMOTE_ENV" ]]; then   # never `[[ ]] && echo` under `set -e`: a false test is the
  echo "      env override: $MU_EVAL_ENV"   # script's last status and would kill the run.
fi

ARGS=""
for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
# ---------------------------------------------------------------------------------------------
# HOLD THE VM-SIDE RECLAIM FOR THE DURATION OF THE RUN.
# ---------------------------------------------------------------------------------------------
# `infra/mu-vm/vm_side_reclaim.sh` runs from the VM's own crontab every 20 minutes and DELETES
# every `mu_mtm__*` collection and every `mu_g__*` graph. Its guard used to be `pgrep -f
# "[p]ytest"` alone — and this harness is not pytest, it is a measurement CLI (see this file's own
# header for why). MEASURED 2026-08-30: the cron fired at 14:20:02 mid-baseline, logged
# `deleted=121 failed=0`, and the run died two seconds later with `Not found: Collection
# mu_mtm__… doesn't exist!` inside `deterministic_promote` — a stack trace that reads like an
# engine defect (ARCHITECTURE-DELTAS AD-201). That guard now also matches `[m]u_eval`, which is
# the primary fix; the hold file below is the belt-and-braces half, so a RENAME of this harness
# cannot silently re-open the hole. Released in a trap, and the reclaim ignores a hold older than
# MU_RECLAIM_HOLD_MAX_S (4 h) so a killed run cannot disable the sweep for ever.
"${SSH[@]}" "touch ~/.mu_reclaim_hold" || true
release_hold() { ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
  "rm -f ~/.mu_reclaim_hold" >/dev/null 2>&1 || true; }
trap release_hold EXIT

echo "[3/3] running: python -m mu_eval$ARGS"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; $REMOTE_ENV \
  cd ~/mu_project/mu-core && PYTHONPATH=eval uv run python -m mu_eval$ARGS"
