#!/bin/bash

#  srun --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=240GB \
#    --gres=gpu:4 --time=0-02:00:00 --qos=normal --account=EUHPC_A02_045 \
#    --job-name=annotate-$PROMPT-$fn-$model_name \
#    --output="${PROJECT}/logs/annotate-${PROMPT}-${model_name}-$fn.log" \
#
#
#  -h, --help            show this help message and exit
#  --model_name MODEL_NAME
#  --language {Swedish,Norwegian,Danish}
#  --num_samples NUM_SAMPLES
#  --batch_size BATCH_SIZE
#  --max_length MAX_LENGTH
#  --max_new_tokens MAX_NEW_TOKENS
#  --cache_dir CACHE_DIR
#  --output_dir OUTPUT_DIR
#  --input_file INPUT_FILE
#  --log_level LOG_LEVEL
#  --log_file LOG_FILE
#  --prompt {askllm,askllm-en,fineweb,fineweb-en}
#
#
#
function run_cmd() {
  PROJECT=$1
  HOME=$2
  MODEL_NAME=$3
  PROMPT=$4
  LANGUAGE=$5
  LANG=$6
  NUM_SAMPLES=$7
  BATCH_SIZE=$8
  MAX_LENGTH=$9
  MAX_NEW_TOKENS=${10}
  CACHE_DIR=${11}
  OUTPUT_DIR=${12}
  LOG_LEVEL=${13}

  echo $PROJECT
  echo $HOME
  echo $MODEL_NAME
  echo $PROMPT
  echo $LANGUAGE
  echo $LANG
  echo $NUM_SAMPLES
  echo $BATCH_SIZE
  echo $MAX_LENGTH
  echo $MAX_NEW_TOKENS
  echo $CACHE_DIR
  echo $OUTPUT_DIR
  echo $LOG_LEVEL

  x=${MODEL_NAME%/*}
  model_name=${x##*/}

  for fn in $(ls $PROJECT/data/dedup-doc-url-gopher/jsonl/$LANG-* | \grep -P "mc4|oscar"); do
    echo $fn
    #srun --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240GB \
    #  --gres=gpu:4 --time=0-24:00:00 --qos=normal --account=EUHPC_A02_045 \
    #  --job-name=annotate-$PROMPT-$fn-$model_name \
    #  --output="${HOME}/logs/annotate-${PROMPT}-${model_name}-$fn.log" \
    #  python $PROJECT/askllm_hf.py \
    #  --model_name $MODEL_NAME \
    #  --language $LANGUAGE \
    #  --num_samples $NUM_SAMPLES \
    #  --batch_size $BATCH_SIZE \
    #  --max_length $MAX_LENGTH \
    #  --max_new_tokens $MAX_NEW_TOKENS \
    #  --cache_dir $CACHE_DIR \
    #  --output_dir $OUTPUT_DIR \
    #  --input_file $fn \
    #  --log_level $LOG_LEVEL \
    #  --prompt $PROMPT &
  done
}

PROJECT="/leonardo_work/EUHPC_A02_045/"
HOME="$PROJECT/scandinavian-lm/robin/"

# AskLLM

MODEL_NAME="$PROJECT/models/Meta-Llama-3.1-8B-Instruct/"
BATCH_SIZE=64
MAX_LENGTH=512
MAX_NEW_TOKENS=5
CACHE_DIR="$HOME/cache_dir/"
OUTPUT_DIR="$HOME/outputs/askllm/"
LOG_LEVEL="info"
PROMPT="askllm"
NUM_SAMPLES=0
#
# SV
LANGUAGE="Swedish"
LANG="sv"
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}

#
# NO
LANGUAGE="Norwegian"
LANG="no"
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
# DA
LANGUAGE="Danish"
LANG="da"
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
#
# Fineweb-Edu - 70b
MODEL_NAME="$PROJECT/models/Meta-Llama-3.1-70B-Instruct/"
BATCH_SIZE=32
MAX_LENGTH=512
MAX_NEW_TOKENS=256
CACHE_DIR="$HOME/cache_dir/"
OUTPUT_DIR="$HOME/outputs/fineweb-70b/"
LOG_LEVEL="info"
PROMPT="fineweb"
#
# SV
LANGUAGE="Swedish"
LANG="sv"
NUM_SAMPLES=12500
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
# NO
LANGUAGE="Norwegian"
LANG="no"
NUM_SAMPLES=27778
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
# DA
LANGUAGE="Danish"
LANG="da"
NUM_SAMPLES=22728
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
#
# Fineweb-Edu - 8b
MODEL_NAME="$PROJECT/models/Meta-Llama-3.1-8B-Instruct/"
BATCH_SIZE=64
MAX_LENGTH=512
MAX_NEW_TOKENS=256
CACHE_DIR="$HOME/cache_dir/"
OUTPUT_DIR="$HOME/outputs/fineweb-8b/"
LOG_LEVEL="info"
PROMPT="fineweb"
#
# SV
LANGUAGE="Swedish"
LANG="sv"
NUM_SAMPLES=25000
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
# NO
LANGUAGE="Norwegian"
LANG="no"
NUM_SAMPLES=55556
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
#
# DA
LANGUAGE="Danish"
LANG="da"
NUM_SAMPLES=45455
run_cmd ${PROJECT} ${HOME} ${MODEL_NAME} ${PROMPT} ${LANGUAGE} ${LANG} \
  ${NUM_SAMPLES} ${BATCH_SIZE} ${MAX_LENGTH} ${MAX_NEW_TOKENS} ${CACHE_DIR} ${OUTPUT_DIR} ${LOG_LEVEL}
