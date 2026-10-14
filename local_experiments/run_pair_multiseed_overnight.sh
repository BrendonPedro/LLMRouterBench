#!/usr/bin/env bash

set -uo pipefail

cd "$HOME/research/LLMRouterBench"
source .venv/bin/activate

BASE_CONFIG="baselines/AvengersPro/config/simple_config_pair_seed42_split0.7.json"

echo "======================================================"
echo "AvengersPro pair multi-seed overnight run"
echo "Seeds: 43-51"
echo "Started: $(date)"
echo "======================================================"

for seed in {43..51}; do

    echo
    echo "######################################################"
    echo "SEED $seed"
    echo "Started: $(date)"
    echo "######################################################"

    DATA_ROOT="baselines/AvengersPro/data/small_models_seed_${seed}"
    SPLIT_DIR="${DATA_ROOT}/seed${seed}_split0.7"

    # --------------------------------------------------
    # 1. Generate the official LLMRouterBench split
    # --------------------------------------------------

    if [[ ! -f "${SPLIT_DIR}/train.jsonl" || \
          ! -f "${SPLIT_DIR}/test.jsonl" || \
          ! -f "${SPLIT_DIR}/baseline_scores.json" ]]; then

        echo "[seed $seed] Creating benchmark split..."

        PYTHONPATH=. python -m baselines.adaptors.avengerspro_adaptor \
          --config config/baseline_config.yaml \
          --seed "$seed" \
          --split-ratio 0.7 \
          --output-dir "$DATA_ROOT"

        status=$?

        if [[ $status -ne 0 ]]; then
            echo "[seed $seed] ADAPTOR FAILED with status $status"
            continue
        fi
    else
        echo "[seed $seed] Split already exists. Reusing it."
    fi


    # --------------------------------------------------
    # 2. Create pair config for this seed
    # --------------------------------------------------

    CONFIG_PATH="baselines/AvengersPro/config/simple_config_pair_seed${seed}_split0.7.json"

    python - "$seed" "$BASE_CONFIG" "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

seed = int(sys.argv[1])
src = Path(sys.argv[2])
dst = Path(sys.argv[3])

with src.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

base = (
    f"baselines/AvengersPro/data/"
    f"small_models_seed_{seed}/seed{seed}_split0.7"
)

cfg["seed"] = seed
cfg["train_data_path"] = f"{base}/train.jsonl"
cfg["test_data_path"] = f"{base}/test.jsonl"
cfg["baseline_scores_path"] = f"{base}/baseline_scores.json"

with dst.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print(f"[seed {seed}] Config saved: {dst}")
PY


    # --------------------------------------------------
    # 3. Run cached two-model AvengersPro
    # --------------------------------------------------

    RESULT_DIR="baselines/AvengersPro/results/pair_seed${seed}_split0.7"
    RESULT_FILE="${RESULT_DIR}/results.json"
    RUN_LOG="local_experiments/results/avengerspro_pair_seed${seed}_run.log"

    mkdir -p "$RESULT_DIR"

    if [[ -f "$RESULT_FILE" ]]; then
        echo "[seed $seed] Result already exists. Skipping."
        continue
    fi

    echo "[seed $seed] Starting AvengersPro..."

    PYTHONPATH=. python local_experiments/run_avengerspro_cached.py \
      --config "$CONFIG_PATH" \
      --output "$RESULT_FILE" \
      --log_level INFO \
      2>&1 | tee "$RUN_LOG"

    status=${PIPESTATUS[0]}

    if [[ $status -ne 0 ]]; then
        echo "[seed $seed] AVENGERSPRO FAILED with status $status"
        continue
    fi

    echo "[seed $seed] Finished successfully: $(date)"

done

echo
echo "======================================================"
echo "OVERNIGHT RUN COMPLETE"
echo "Finished: $(date)"
echo "======================================================"
