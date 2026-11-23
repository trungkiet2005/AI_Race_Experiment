#!/bin/bash
# Collect one declared slice of docs/ijcai-revision-collection-plan-2026-11-23.md
# on its declared identity: bake, push, run every assigned route, download, ingest.
# Usage: run_ijcai_revision_slice.sh IDENTITY STRATEGY RISK BLOCK(A|B|C) GROUP(admitted|refused)
set -u
IDENT="$1"; STRAT="$2"; RISK="$3"; BLOCK="$4"; GROUP="$5"
[ -n "$IDENT" ] && [ -n "$STRAT" ] && [ -n "$RISK" ] && [ -n "$BLOCK" ] && [ -n "$GROUP" ] || { echo "missing argument"; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
POOL="$HOME/PhD_Farming/infra/kaggle_for_research/scripts/kaggle_pool.py"
PLAN="docs/ijcai-revision-collection-plan-2026-11-23.md"
TASK="ai-race-scripted-opponent"
WORK="$REPO/results/_build/ijcai_revision/${BLOCK}_${STRAT}_${RISK}_${IDENT}_$(date +%Y%m%d%H%M%S)"
mkdir -p "$WORK"
LOG="$WORK/log.txt"
exec > >(tee -a "$LOG") 2>&1

case "$BLOCK" in
  A) SEED=260726; SEEDARG="" ;;
  B) SEED=270726; SEEDARG="--base-seed 270726" ;;
  C) SEED=280726; SEEDARG="--base-seed 280726" ;;
  *) echo "bad block"; exit 2 ;;
esac
EXTRA_BAKE=""
case "$GROUP" in
  admitted) MODELS="gemini-3-flash-preview gpt-5.4-2026-03-05 gpt-5.5-2026-04-23 claude-sonnet-5-default claude-opus-5-default"
            if [ "$BLOCK" = "A" ]; then echo "admitted routes use blocks B/C"; exit 2; fi
            CAMPAIGN="results/frontier/scripted_opponent_replication/block_${BLOCK}" ;;
  refused)  MODELS="gemini-3.1-flash-lite-preview gpt-5.4-mini-2026-03-17 gemini-3.5-flash-lite gpt-5.4-nano-2026-03-17"
            CAMPAIGN="results/frontier/scripted_opponent_refused" ;;
  swap_sonnet) MODELS="claude-sonnet-5-default"; EXTRA_BAKE="--add-reasoning-low claude-sonnet-5"
            CAMPAIGN="results/frontier/scripted_opponent_contract_swap/sonnet_low" ;;
  harmonize_opus) MODELS="claude-opus-5-default"
            CAMPAIGN="results/frontier/scripted_opponent_contract_swap/opus_active" ;;
  *) echo "bad group"; exit 2 ;;
esac

route_of() {
  case "$1" in
    claude-*-default) echo "anthropic/${1%-default}@default" ;;
    gemini-*) echo "google/$1" ;;
    gpt-*) echo "openai/$1" ;;
  esac
}
pool() { python3 "$POOL" run "$IDENT" -- "$@"; }

echo "== $(date -u +%FT%TZ) slice $BLOCK $STRAT $RISK on $IDENT -> $CAMPAIGN"
cd "$REPO" || exit 2
python3 scripts/bake_scripted_opponent_cell.py --strategy "$STRAT" --risk "$RISK" $SEEDARG $EXTRA_BAKE \
  --output "$WORK/ai_race_scripted_opponent.py" || exit 3
shasum -a 256 "$WORK/ai_race_scripted_opponent.py"

# Long operations are never awaited inside one CLI call (the pool wrapper kills a
# call after 900 s); they are started and then polled.
task_status() { pool b t status "$TASK" 2>&1 | awk '/^Status:/{print $2; exit}'; }
route_status() { pool b t status "$TASK" -m "$1" 2>&1 | awk -v m="$1" '$1==m{print $2; exit}'; }
wait_task() {
  local i s
  for i in $(seq 1 240); do
    s="$(task_status)"
    case "$s" in Completed) return 0 ;; Errored|Failed|ERRORED) return 1 ;; esac
    sleep 60
  done
  return 1
}

if [ "${SKIP_PUSH:-0}" != "1" ]; then
  echo "== push $(date -u +%T)"
  OUT="$(pool b t push "$TASK" -f "$WORK/ai_race_scripted_opponent.py" 2>&1)"; echo "$OUT" | tail -5
  echo "$OUT" | grep -q -i -E "exceeds your available quota|identity verification required" && { echo "PUSH_FAILED"; exit 4; }
else
  echo "== push skipped (resume)"
fi
if ! wait_task; then
  pool b t status "$TASK" 2>&1 | tail -20
  # The status view hides the provider error; read it from the validation run.
  VDL="$WORK/validation_dl"; mkdir -p "$VDL"
  VM="$(pool b t status "$TASK" 2>&1 | awk '$1!="Status:" && $2=="Errored"{print $1; exit}')"
  [ -n "$VM" ] && pool b t download "$TASK" -m "$VM" -o "$VDL" >/dev/null 2>&1
  grep -rho -E "exceeds your available quota|identity verification required" "$VDL" 2>/dev/null | sort -u
  echo "PUSH_FAILED"; exit 4
fi
echo "== push validated $(date -u +%T)"

if [ "${SKIP_RUN:-0}" != "1" ]; then
  RUNARGS=""
  for m in $MODELS; do RUNARGS="$RUNARGS -m $m"; done
  echo "== run $(date -u +%T): $MODELS"
  pool b t run "$TASK" $RUNARGS 2>&1 | tail -3
else
  echo "== run skipped (resume)"
fi
for m in $MODELS; do
  for i in $(seq 1 360); do
    s="$(route_status "$m")"
    case "$s" in Running|Queued|Pending|"") sleep 60 ;; *) break ;; esac
  done
  echo "== route $m final status: ${s:-unknown} $(date -u +%T)"
done

for m in $MODELS; do
  echo "== collect $m $(date -u +%T)"
  DL="$WORK/dl_$m"
  mkdir -p "$DL"
  pool b t download "$TASK" -m "$m" -o "$DL" >/dev/null 2>&1 || { echo "DOWNLOAD_FAILED $m"; continue; }
  python3 scripts/ingest_scripted_cell.py --download-root "$DL" --identity "$IDENT" \
    --plan "$PLAN" --campaign "$CAMPAIGN" --expect-route "$(route_of "$m")" \
    --expect-base-seed "$SEED" || echo "INGEST_REFUSED $m"
done
echo "== done $(date -u +%FT%TZ)"
