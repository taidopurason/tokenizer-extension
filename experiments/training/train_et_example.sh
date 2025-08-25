#!/bin/bash

DATASET_NAME=fineweb
LANG=ekk_Latn
BUDGET=1000000000

SECONDS=0
bash sp_training.sh ${DATASET_NAME} ${LANG} "llama2" "meta-llama/Llama-2-7b-hf" ${BUDGET} "/gpfs/helios/home/taido/projects/tokenizer-extension/sp_models/llama-2/tokenizer.model"
bash hf_training.sh ${DATASET_NAME} ${LANG} "llama3" "meta-llama/Meta-Llama-3.1-8B" ${BUDGET}
bash hf_training.sh ${DATASET_NAME} ${LANG} "mistralnemo" "mistralai/Mistral-Nemo-Base-2407" ${BUDGET}
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."