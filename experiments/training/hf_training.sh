dataset_name=$1
lang=$2
model_name=$3
model_path=$4
budget=$5

DATA_DIR="/gpfs/helios/home/taido/projects/tokenizer-extension/datasets"
SCRIPT_DIR="/gpfs/helios/home/taido/projects/tokenizer-extension/scripts"
OUT_DIR="/gpfs/helios/home/taido/projects/tokenizer-extension/results/exp_2"
mkdir -p ${OUT_DIR}

EXT_OUT_PATH="${OUT_DIR}/${dataset_name}-${lang}/extension-${model_name}-${budget}"
mkdir -p ${EXT_OUT_PATH}
python -u ${SCRIPT_DIR}/train_extension.py \
  --input_path="${DATA_DIR}/${dataset_name}-${lang}/${budget}_hf" \
  --output_path="${EXT_OUT_PATH}" \
  --tokenizer_path=${model_path} \
  --vocab_size=64000

for size in 1000 2000 4000 8000 16000 32000 64000; do
  python -u ${SCRIPT_DIR}/extend_tokenizer.py \
    --tokenizer_path=${model_path} \
    --output_path="${OUT_DIR}/${dataset_name}-${lang}/tokenizers/extension-${model_name}-${budget}-ext${size}" \
    --n_tokens=${size} \
    --extension_method="continued-training" \
    --extension_path="${EXT_OUT_PATH}"\
    --keep_added_token_positions=False
done

FS_OUT_PATH="${OUT_DIR}/${dataset_name}-${lang}/fromscratch-${model_name}-${budget}/tokenizer"
mkdir -p ${FS_OUT_PATH}
python -u ${SCRIPT_DIR}/train_from_scratch.py \
  --input_path="${DATA_DIR}/${dataset_name}-${lang}/${budget}_hf" \
  --output_path="${FS_OUT_PATH}" \
  --tokenizer_path=${model_path} \
  --vocab_size=64000


for size in 1000 2000 4000 8000 16000 32000; do
  python -u ${SCRIPT_DIR}/extend_tokenizer.py \
    --tokenizer_path=${model_path} \
    --output_path="${OUT_DIR}/${dataset_name}-${lang}/tokenizers/fromscratch-${model_name}-${budget}-ext${size}" \
    --n_tokens=${size} \
    --extension_method="hf-model" \
    --extension_path="${FS_OUT_PATH}" \
    --keep_added_token_positions=False
done
