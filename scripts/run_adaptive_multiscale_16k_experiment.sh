#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_CONFIG="${RUN_CONFIG:-$ROOT/configs/adaptive_multiscale_16k.env}"
MODE="${1:-check}"

case "$MODE" in
  check)
    bash "$ROOT/scripts/train_distributed.sh" check
    ;;
  audit)
    bash "$ROOT/scripts/train_distributed.sh" audit
    ;;
  smoke)
    bash "$ROOT/scripts/train_distributed.sh" smoke
    ;;
  train)
    bash "$ROOT/scripts/train_distributed.sh" train
    ;;
  evaluate)
    bash "$ROOT/scripts/evaluate_adaptive_multiscale_balanced100.sh" run
    ;;
  all)
    bash "$ROOT/scripts/train_distributed.sh" audit
    bash "$ROOT/scripts/train_distributed.sh" check
    bash "$ROOT/scripts/train_distributed.sh" smoke
    bash "$ROOT/scripts/train_distributed.sh" train
    bash "$ROOT/scripts/evaluate_adaptive_multiscale_balanced100.sh" run
    ;;
  status)
    bash "$ROOT/scripts/train_distributed.sh" status
    bash "$ROOT/scripts/evaluate_adaptive_multiscale_balanced100.sh" status
    ;;
  *) echo "Usage: $0 {audit|check|smoke|train|evaluate|all|status}" >&2; exit 2 ;;
esac
