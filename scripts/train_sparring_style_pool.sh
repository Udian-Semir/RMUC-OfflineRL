#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniconda3/envs/nerfstudio/bin/python}"
ROOT="${1:-sparring_style_pool}"

for style in aggressive pressure measured; do
  for role in hero engineer infantry aerial sentry; do
    config="configs/${role}_iql_tactical_buildings.yaml"
    run_role="$role"
    if [[ "$role" == "sentry" ]]; then
      config="configs/sentry_iql_tactical_buildings.yaml"
      run_role="blue_sentry"
    fi
    data="$ROOT/datasets/$style/$role"
    out="$ROOT/checkpoints/$style/$role"
    "$PYTHON" -m rm_rl.data.check_dataset --data "$data" | tee "$out/check_log.txt"
    "$PYTHON" -m rm_rl.train.train_offline --config "$config" --data "$data" --out "$out" | tee "$out/train_stdout.txt"
    "$PYTHON" -m rm_rl.eval.write_role_report --data "$data" --run "$out" --role "${style}_${run_role}"
  done
done
