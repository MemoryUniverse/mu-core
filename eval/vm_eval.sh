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
  # MU_EVAL_ENV is a generic engine-knob passthrough (MU_RECALL__*, MU_LIFECYCLE__*, …), never a
  # secret seam — it is echoed one line above AND embedded verbatim into the remote command string
  # below, exactly the two places a real API key must never appear (see the --api-key fix right
  # below this block). A loud warning, not a hard block: this override predates the fix and this
  # script cannot know every future knob name, so a name that LOOKS like a secret gets flagged
  # rather than silently trusted.
  if [[ "$MU_EVAL_ENV" =~ (KEY|SECRET|PASSWORD|TOKEN) ]]; then
    echo "      !! MU_EVAL_ENV looks like it may carry a secret (matched KEY/SECRET/PASSWORD/" >&2
    echo "         TOKEN) — this value is echoed above and reaches the VM's process table for" >&2
    echo "         the whole run. Use MU_EVAL_API_KEY for the Azure/OpenAI-compatible key" >&2
    echo "         instead (see this script's --api-key comment)." >&2
  fi
fi

# ---------------------------------------------------------------------------------------------
# THE API KEY — OUT OF BAND, NEVER IN ARGV OR A LOG.
# ---------------------------------------------------------------------------------------------
# SECURITY FIX (verified live via `pgrep` on the VM): `--api-key <value>` used to be passed
# through to `mu_eval` like any other CLI arg — `printf %q`-quoted into `$ARGS` below, which is
# BOTH embedded in the `ssh ... "... python -m mu_eval$ARGS"` command string (putting the key in
# THIS process's own argv for the whole run — `ps`/`/proc/<pid>/cmdline`, world-readable on a
# shared box) AND, on the far side, in the `python -m mu_eval` process's own argv in the VM's
# process table, AND in the `echo "[3/3] running: ...$ARGS"` line just below (redirected to a log
# by any caller that does `vm_eval.sh ... > run.log`). None of that is where a secret belongs
# (DEV-STANDARDS: "secrets only from the secret seam — never in logs, errors, commits, or config
# literals").
#
# The safe path: `export MU_EVAL_API_KEY=...` locally (never pass `--api-key` at all) — this
# script picks it up from ITS OWN environment (never echoed, never put in argv here either),
# writes it to a mode-600 local temp file, ships THAT file to a mode-600, per-run-unique path on
# the VM (rsync, never scp -B/cat-through-ssh — no intermediate shell ever sees the byte content
# as a command-line argument), and the remote command reads it back with `export
# MU_EVAL_API_KEY="$(cat <path>)"` — a `cat` of a FILENAME, never the secret, appears in any argv.
# `mu_eval/__main__.py`'s own `_resolve_api_key` then prefers this env var over `--api-key`.
#
# `--api-key` on THIS script's own command line still works (back-compat / a quick one-off run)
# — stripped out of `$ARGS` below so it never reaches the log/remote-argv paths above, but note
# that it is still visible in the LOCAL shell history and in `ps` for this script's own
# invocation, which `MU_EVAL_API_KEY` is not. `mu_eval` itself also warns when it receives a key
# via that path (`_resolve_api_key`'s own message).
API_KEY="${MU_EVAL_API_KEY:-}"
ARGS=""
skip_next=0
for a in "$@"; do
  if [[ "$skip_next" == 1 ]]; then
    skip_next=0
    if [[ -z "$API_KEY" ]]; then
      API_KEY="$a"
      echo "      ! --api-key was passed on vm_eval.sh's own command line — prefer" >&2
      echo "        'export MU_EVAL_API_KEY=...' instead (never appears in argv/logs)." >&2
    fi
    continue
  fi
  case "$a" in
    --api-key)
      skip_next=1
      continue
      ;;
    --api-key=*)
      if [[ -z "$API_KEY" ]]; then
        API_KEY="${a#--api-key=}"
        echo "      ! --api-key was passed on vm_eval.sh's own command line — prefer" >&2
        echo "        'export MU_EVAL_API_KEY=...' instead (never appears in argv/logs)." >&2
      fi
      continue
      ;;
  esac
  ARGS+=" $(printf '%q' "$a")"
done

# shellcheck disable=SC2088  # deliberate: expanded by the REMOTE shell everywhere this is used
# (rsync's `user@host:path` destination, an ssh command string) — same pattern this file already
# uses unflagged for ~/mu_project/mu-core and ~/mu_eval_data above; never expanded locally.
REMOTE_KEY_PATH="~/.mu_eval_api_key.$$"
KEY_TMP=""
if [[ -n "$API_KEY" ]]; then
  KEY_TMP="$(umask 077 && mktemp)"
  printf '%s' "$API_KEY" > "$KEY_TMP"
  chmod 600 "$KEY_TMP"
  rsync -az -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "$KEY_TMP" "user@$IP:$REMOTE_KEY_PATH"
  "${SSH[@]}" "chmod 600 $REMOTE_KEY_PATH"
  REMOTE_ENV="export MU_EVAL_API_KEY=\"\$(cat $REMOTE_KEY_PATH)\"; $REMOTE_ENV"
fi
cleanup_key() {
  [[ -n "$KEY_TMP" ]] && rm -f "$KEY_TMP"
  # Skip the remote round-trip entirely when no key was ever shipped (the common case — most
  # subcommands need none) rather than deleting a path that was never created.
  if [[ -n "$KEY_TMP" ]]; then
    ssh -n -i "$SSH_KEY" -o StrictHostKeyChecking=no "user@$IP" \
      "rm -f $REMOTE_KEY_PATH" >/dev/null 2>&1 || true
  fi
}

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
trap 'release_hold; cleanup_key' EXIT

echo "[3/3] running: python -m mu_eval$ARGS"
"${SSH[@]}" "export PATH=\$HOME/.local/bin:\$PATH; export MU_ON_VM=1; $REMOTE_ENV \
  cd ~/mu_project/mu-core && PYTHONPATH=eval uv run python -m mu_eval$ARGS"
