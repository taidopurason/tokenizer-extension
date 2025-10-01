GPUS_PER_NODE=8
LR_START=1e-4
EVAL_FREQ=256
SAVE_FREQ=512
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=$TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION=4
MAX_SEQ_LEN=4096
TOTAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GPUS_PER_NODE * GRADIENT_ACCUMULATION * SLURM_NNODES))
echo "Total batch size: ${TOTAL_BATCH_SIZE} Training samples: ${TRAIN_SAMPLES}; Max steps: ${MAX_STEPS}; Total tokens: ${TOTAL_TOKENS}"
echo "Warmup steps ${WARMUP_STEPS}"

DATA_PATH="data/packed-4096/llama3"
TRAIN_PATH="${DATA_PATH}/fineweb-ekk_Latn,${DATA_PATH}/fineweb-eng_Latn"

VALID_PATH="ekk_Latn:tokenizers_fw_validation:ekk_Latn,eng_Latn:tokenizers_fw_validation:eng_Latn"
OUTPUT_DIR=checkpoints/${RUN_NAME}

mkdir -p ${OUTPUT_DIR}
PRE_TRAINED_MODEL="meta-llama/Llama-3.2-3B"

export LAUNCHER="accelerate launch \
  --config_file ${WORKING_DIR}/fsdp_train_config_gradop.yaml \
  --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
  --num_machines ${SLURM_NNODES} \
  --rdzv_backend c10d \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision bf16 \
"
export PYTHON_FILE="continued_pretraining.py"
export ARGS="--model_name ${PRE_TRAINED_MODEL} \
    --run_name ${RUN_NAME} \
    --tokenizer_name ${PRE_TRAINED_MODEL} \
    --low_cpu_mem_usage \
    --valid_split_name validation \
    --train_path ${TRAIN_PATH} \
    --valid_path ${VALID_PATH} \
    --train_dataset_type huggingface_local_packed \
    --valid_dataset_type huggingface \
    --eval_packing True \
    --dataset_text_field text \
    --report_to wandb \
    --seed 42 \
    --max_seq_len ${MAX_SEQ_LEN} \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --eval_steps ${EVAL_FREQ} \
    --save_steps ${SAVE_FREQ} \
    --eval_strategy steps \
    --save_strategy steps \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_final_model \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LR_START} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs {\"min_lr_rate\":0.1} \
    --bf16 True \
    --weight_decay 0.1 \
    --torch_dtype bfloat16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --use_flash_attention_2 \
    --pad_token <|finetune_right_pad_id|>"

export CMD="$LAUNCHER $PYTHON_FILE $ARGS"

echo "Running $CMD"


echo "Running script"
srun \
 singularity exec $SIF ${WORKING_DIR}/runscript.sh ${CMD}

