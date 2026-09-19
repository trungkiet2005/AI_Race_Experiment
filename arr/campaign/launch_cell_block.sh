#!/usr/bin/env bash
# Push one family task on one identity and run its preregistered routes.
# Usage: launch_cell_block.sh IDENTITY TASK_SLUG TASK_FILE LOG route [route ...]
# Assignment is fixed in docs/arr-transfer-campaign-prereg-2026-09-19.md, section 5.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
POOL="D:/PhD_LetGoo/PhD_Farming/infra/kaggle_for_research/scripts/kaggle_pool.py"
IDENTITY="$1"; SLUG="$2"; TASK_FILE="$3"; LOG="$4"; shift 4
{
  echo "== $(date -u +%FT%TZ) push $SLUG on $IDENTITY"
  python "$POOL" run "$IDENTITY" b t push "$SLUG" -f "$TASK_FILE" --wait 1800
  echo "== $(date -u +%FT%TZ) run $SLUG on $IDENTITY: $*"
  ARGS=()
  for route in "$@"; do ARGS+=(-m "$route"); done
  python "$POOL" run "$IDENTITY" b t run "$SLUG" "${ARGS[@]}" --wait 7200
  echo "== $(date -u +%FT%TZ) status"
  python "$POOL" run "$IDENTITY" b t status "$SLUG"
  echo "== $(date -u +%FT%TZ) done"
} >> "$LOG" 2>&1
