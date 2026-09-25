#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
if [ -e project ] || [ -e data ] || [ -e run ]; then
  echo "Use a fresh transfer directory; existing project/data/run found." >&2
  exit 1
fi
mkdir project
cd project
git init -q
git fetch -q ../source.bundle HEAD
git checkout -q --detach FETCH_HEAD
cd ..
mkdir data
tar -xzf data.tar.gz -C data
"$PYTHON" project/tools/server_preflight_j02.py --protocol project/nodes/J-02/run-protocol.json --output server-protocol.json --gpu "$GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
finish_export() {
  status=$?
  trap - EXIT
  if [ -d run ]; then
    mkdir -p run/reproduction-inputs
    cp source.bundle data.tar.gz server-protocol.json run/reproduction-inputs/
    if [ -f training-console.log ]; then cp training-console.log run/training-console.log; fi
    "$PYTHON" project/tools/export_training_run.py --run "$PWD/run" --output "$PWD/training-export.zip" || status=$?
  fi
  exit "$status"
}
trap finish_export EXIT
set +e
"$PYTHON" project/tools/run_j02_development.py --data "$PWD/data" --output "$PWD/run" --protocol "$PWD/server-protocol.json" > training-console.log 2>&1
run_status=$?
set -e
if [ "$run_status" -eq 0 ]; then
  "$PYTHON" project/tools/evaluate_j02_gates.py --run "$PWD/run" --output "$PWD/run/gates.json"
fi
exit "$run_status"
