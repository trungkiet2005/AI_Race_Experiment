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
case "$GROUP" in
  admitted) MODELS="gemini-3-flash-preview gpt-5.4-2026-03-05 gpt-5.5-2026-04-23 claude-sonnet-5-default claude-opus-5-default"
            if [ "$BLOCK" = "A" ]; then echo "admitted routes use blocks B/C"; exit 2; fi
            CAMPAIGN="results/frontier/scripted_opponent_replication/block_${BLOCK}" ;;
  refused)  MODELS="gemini-3.1-flash-lite-preview gpt-5.4-mini-2026-03-17 gemini-3.5-flash-lite gpt-5.4-nano-2026-03-17"
            CAMPAIGN="results/frontier/scripted_opponent_refused" ;;
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
python3 scripts/bake_scripted_opponent_cell.py --strategy "$STRAT" --risk "$RISK" $SEEDARG \
  --output "$WORK/ai_race_scripted_opponent.py" || exit 3
shasum -a 256 "$WORK/ai_race_scripted_opponent.py"

echo "== push $(date -u +%T)"
pool b t push "$TASK" -f "$WORK/ai_race_scripted_opponent.py" --wait || { echo "PUSH_FAILED"; exit 4; }

RUNARGS=""
for m in $MODELS; do RUNARGS="$RUNARGS -m $m"; done
echo "== run $(date -u +%T): $MODELS"
pool b t run "$TASK" $RUNARGS --wait || echo "RUN_RETURNED_NONZERO (per-route status below)"

for m in $MODELS; do
  echo "== route $m $(date -u +%T)"
  pool b t status "$TASK" -m "$m" 2>&1 | tail -3
  DL="$WORK/dl_$m"
  mkdir -p "$DL"
  pool b t download "$TASK" -m "$m" -o "$DL" >/dev/null 2>&1 || { echo "DOWNLOAD_FAILED $m"; continue; }
  python3 scripts/ingest_scripted_cell.py --download-root "$DL" --identity "$IDENT" \
    --plan "$PLAN" --campaign "$CAMPAIGN" --expect-route "$(route_of "$m")" \
    --expect-base-seed "$SEED" || echo "INGEST_REFUSED $m"
done
echo "== done $(date -u +%FT%TZ)"
