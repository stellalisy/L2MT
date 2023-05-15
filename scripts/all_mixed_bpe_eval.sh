#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N all-mixed-eval-sacrebleu
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=octopod
#$ -q g.q
#$ -t 1
#$ -j y -o /home/sli136/l2mt/output/bpe-mt/

# b1[123456789]|c0*|c1[123456789]

source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/l2mt

########################################################
# 0. Define variables
########################################################
KEY=all-mixed
rawfile_key='all-mixed'                                             # name identifier for raw data in the format "train.${rawfile_key}.en"
RAW_SOURCE='/home/sli136/l2mt/data/raw/'${rawfile_key}              # raw (untokenized) data (artificial mixed data directory, doesn't include l2 and ref data)
DEST_DIR='/export/c11/sli136/l2mt/data/input_data_'${rawfile_key}   # parent directory for all tokenized data to store input for training
BPE_DIR=${DEST_DIR}'/bpe_tokenized'                                 # bpe tokenized data for fairseq training
BASE_BIN=${DEST_DIR}'/base_bin'                                     # fairseq binarized data for fairseq training
L2_BIN=${DEST_DIR}'/l2_bin'                                         # fairseq binarized data for l2 eval (original crappy sentences)
REF_BIN=${DEST_DIR}'/ref_bin'                                       # fairseq binarized data for ref eval (human corrected sentences)
MODEL_DIR='/export/c11/sli136/l2mt/models/'${KEY}                   # parent directory for all models to store output from training

[ ! -d ${DEST_DIR} ] && mkdir -p ${DEST_DIR}
[ ! -d ${BPE_DIR} ] && mkdir -p ${BPE_DIR}
[ ! -d ${BASE_BIN} ] && mkdir -p ${BASE_BIN}
[ ! -d ${REF_BIN} ] && mkdir -p ${REF_BIN}
[ ! -d ${L2_BIN} ] && mkdir -p ${L2_BIN}
[ ! -d ${MODEL_DIR} ] && mkdir -p ${MODEL_DIR}

SRC='en'
TGT='es'

BPE='--bpe sentencepiece --sentencepiece-model '${MODEL_DIR}'/bpe.model'
LOG_FILE=${MODEL_DIR}/log.out

########################################################
# 4. Fairseq evaluate
########################################################
# echo "================evaluating ref (corrected) data bleu================"
# RESULT_PATH=${MODEL_DIR}'/results/ref/bleu/'
# [ ! -d ${RESULT_PATH} ] && mkdir -p ${RESULT_PATH}
# fairseq-generate ${REF_BIN} --path $MODEL_DIR/checkpoint_best.pt \
#     ${BPE} \
#     --task translation \
#     --beam 5 --lenpen 1.0 \
#     --batch-size 16  \
#     --scoring bleu \
#     --sacrebleu \
#     --source-lang ${SRC} --target-lang ${TGT} \
#     --results-path ${RESULT_PATH}
echo "================evaluating ref (corrected) data sacrebleu with sacrebleu================"
RESULT_PATH=${MODEL_DIR}'/results/ref/sacrebleu/'
[ ! -d ${RESULT_PATH} ] && mkdir -p ${RESULT_PATH}
fairseq-generate ${REF_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --scoring sacrebleu \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}

echo "================evaluating ref (corrected) data sacrebleu withouth sacrebleu================"
RESULT_PATH=${MODEL_DIR}'/results/ref/sacrebleu/'
[ ! -d ${RESULT_PATH} ] && mkdir -p ${RESULT_PATH}
fairseq-generate ${REF_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --scoring sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
# python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} --sacrebleu --bleu

# echo "================evaluating l2 (crappy) data bleu================"
# RESULT_PATH=${MODEL_DIR}'/results/l2/bleu'
# [ ! -d ${RESULT_PATH} ] && mkdir -p ${RESULT_PATH}
# fairseq-generate ${L2_BIN} --path $MODEL_DIR/checkpoint_best.pt \
#     ${BPE} \
#     --task translation \
#     --beam 5 --lenpen 1.0 \
#     --batch-size 16  \
#     --scoring bleu \
#     --sacrebleu \
#     --source-lang ${SRC} --target-lang ${TGT} \
#     --results-path ${RESULT_PATH}

# echo "================evaluating l2 (crappy) data sacrebleu================"
# RESULT_PATH=${MODEL_DIR}'/results/l2/sacrebleu'
# [ ! -d ${RESULT_PATH} ] && mkdir -p ${RESULT_PATH}
# fairseq-generate ${L2_BIN} --path $MODEL_DIR/checkpoint_best.pt \
#     ${BPE} \
#     --task translation \
#     --beam 5 --lenpen 1.0 \
#     --batch-size 16  \
#     --scoring sacrebleu \
#     --sacrebleu \
#     --source-lang ${SRC} --target-lang ${TGT} \
#     --results-path ${RESULT_PATH}
# python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} --sacrebleu --bleu
