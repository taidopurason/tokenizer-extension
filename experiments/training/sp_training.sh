dataset_name=$1
lang=$2
model_name=$3
model_path=$4
budget=$5
pretrained_model_path=$6

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
  --vocab_size=64000 \
  --is_sentencepiece=True \
  --nfkc_normalize=True \
  --required_sp_coverage=0.99995 \
  --max_token_length=16

for size in 1000 2000 4000 8000 16000 32000 64000; do
  python -u ${SCRIPT_DIR}/extend_tokenizer.py \
    --tokenizer_path=${model_path} \
    --output_path="${OUT_DIR}/${dataset_name}-${lang}/tokenizers/extension-${model_name}-${budget}-ext${size}" \
    --n_tokens=${size} \
    --extension_method="continued-training" \
    --extension_path="${EXT_OUT_PATH}"\
    --keep_added_token_positions=False \
    --is_sentencepiece=True
done


FS_OUT_PATH="${OUT_DIR}/${dataset_name}-${lang}/fromscratch-${model_name}-${budget}"
mkdir -p ${FS_OUT_PATH}
python -u ${SCRIPT_DIR}/train_from_scratch.py \
  --tokenizer_path=${pretrained_model_path} \
  --input_path="${DATA_DIR}/${dataset_name}-${lang}/${budget}_lines/train_data.txt" \
  --output_path="${FS_OUT_PATH}/tokenizer" \
  --vocab_size=64000 \
  --sp_num_threads=16 \
  --implementation="sentencepiece"


for size in 1000 2000 4000 8000 16000 32000; do
  python -u ${SCRIPT_DIR}/extend_tokenizer.py \
    --tokenizer_path=${model_path} \
    --output_path="${OUT_DIR}/${dataset_name}-${lang}/tokenizers/fromscratch-${model_name}-${budget}-ext${size}" \
    --n_tokens=${size} \
    --extension_method="sp-model" \
    --extension_path="${FS_OUT_PATH}/tokenizer.vocab" \
    --keep_added_token_positions=False \
    --is_sentencepiece=True \
    --sp_add_chars_first=True
done