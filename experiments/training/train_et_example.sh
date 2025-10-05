#!/bin/bash

DATASET_NAME=fineweb
LANG=ekk_Latn
BUDGET=1000000000
DATA_DIR=../datasets
OUT_DIR=../results

SECONDS=0
bash sp_training.sh ${DATASET_NAME} ${LANG} "llama2" "meta-llama/Llama-2-7b-hf" ${BUDGET} "sp_models/llama-2/tokenizer.model" ${DATA_DIR} ${OUT_DIR}
bash hf_training.sh ${DATASET_NAME} ${LANG} "llama3" "meta-llama/Meta-Llama-3.1-8B" ${BUDGET} ${DATA_DIR} ${OUT_DIR}
bash hf_training.sh ${DATASET_NAME} ${LANG} "mistralnemo" "mistralai/Mistral-Nemo-Base-2407" ${BUDGET} ${DATA_DIR} ${OUT_DIR}
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."