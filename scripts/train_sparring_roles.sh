#!/usr/bin/env bash
# Build, validate and train every non-red-sentry role used by the sparring pool.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniconda3/envs/nerfstudio/bin/python}"
DB="${DB:-dataset/rmuc_2026_region_dataset.sqlite}"
ROLES=(hero engineer aerial sentry)

for role in "${ROLES[@]}"; do
  config="configs/${role}_iql_tactical.yaml"
  data="data/${role}_tactical"
  run="rm_runs/${role}_iql_tactical"
  if [[ "$role" == "sentry" ]]; then
    config="configs/blue_sentry_iql_tactical.yaml"
    run="rm_runs/blue_sentry_iql_tactical"
  fi
  mkdir -p "$data" "$run"
  "$PYTHON" -m rm_rl.data.build_dataset \
    --db "$DB" --out "$data" --agent "$role" --config "$config" \
    --action-mode tactical --goal-horizon 5 --match-seconds 420 \
    --vis-map data/vis_map.npz --team-prior data/team_prior.json \
    | tee "$data/build_log.txt"
  "$PYTHON" -m rm_rl.data.check_dataset --data "$data" | tee "$data/check_log.txt"
  "$PYTHON" -m rm_rl.train.train_offline --config "$config" | tee "$run/train_stdout.txt"
  "$PYTHON" -m rm_rl.eval.write_role_report --data "$data" --run "$run" --role "$role"
done
