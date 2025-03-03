#!/usr/bin/env bash

MODEL_PATH="model/LLaGA/checkpoints"
MODEL_BASE="model/LLaGA/vicuna-7b-v1.5-16k"  # or 'meta-llama/Llama-2-7b-hf'
MODE="v1"            # use 'llaga_llama_2' for llama, or "v1" for others
DATASET="cora_sup"   # test dataset
PROMPT_FILE="model/LLaGA/dataset/sup/cora_sup/sampled_2_10_test_prbcd_local.jsonl"
TASK="nc"            # test task
EMB="sbert"
USE_HOP=2            # 2 for ND, 4 for HO
SAMPLE_SIZE=10
TEMPLATE="ND"

OUTPUT_PATH="model/LLaGA/output/${DATASET}/${DATASET}_${TASK}_${TEMPLATE}_prbcd_l.txt"

python model/LLaGA/eval/eval_pretrain.py \
    --model_path "${MODEL_PATH}" \
    --model_base "${MODEL_BASE}" \
    --conv_mode  "${MODE}" \
    --dataset "${DATASET}" \
    --pretrained_embedding_type "${EMB}" \
    --use_hop "${USE_HOP}" \
    --sample_neighbor_size "${SAMPLE_SIZE}" \
    --answers_file "${OUTPUT_PATH}" \
    --task "${TASK}" \
    --cache_dir \
    --template "${TEMPLATE}" \
    --prompt_file "${PROMPT_FILE}"
