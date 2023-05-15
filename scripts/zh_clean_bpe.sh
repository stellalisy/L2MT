#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N zh-clean
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

tokenize=${1:-"1"}
preprocess=${2:-"1"}
train=${3:-"1"}
eval=${4:-"1"}

########################################################
# 0. Define variables
########################################################
SRC='en'
TGT='zh'

KEY=clean
rawfile_key='clean'                                        # name identifier for raw data in the format "train.${rawfile_key}.${SRC}"
RAW_SOURCE=/export/c11/sli136/l2mt/${SRC}-${TGT}/data/raw/${rawfile_key}      # raw (untokenized) data (artificial mixed data directory, doesn't include l2 and ref data)
DEST_DIR=/export/c11/sli136/l2mt/${SRC}-${TGT}/data/input_data_${KEY}       # parent directory for all tokenized data to store input for training
REF_L2_DIR=/export/c11/sli136/l2mt/${SRC}-${TGT}/data/raw
BPE_DIR=${DEST_DIR}'/bpe_tokenized'                        # bpe tokenized data for fairseq training
BASE_BIN=${DEST_DIR}'/base_bin'                            # fairseq binarized data for fairseq training
L2_BIN=${DEST_DIR}'/l2_bin'                                # fairseq binarized data for l2 eval (original crappy sentences)
REF_BIN=${DEST_DIR}'/ref_bin'                              # fairseq binarized data for ref eval (human corrected sentences)
MODEL_DIR=/export/c11/sli136/l2mt/${SRC}-${TGT}/models/${KEY}          # parent directory for all models to store output from training

[ ! -d ${DEST_DIR} ] && mkdir -p ${DEST_DIR}
[ ! -d ${BPE_DIR} ] && mkdir -p ${BPE_DIR}
[ ! -d ${BASE_BIN} ] && mkdir -p ${BASE_BIN}
[ ! -d ${REF_BIN} ] && mkdir -p ${REF_BIN}
[ ! -d ${L2_BIN} ] && mkdir -p ${L2_BIN}
[ ! -d ${MODEL_DIR} ] && mkdir -p ${MODEL_DIR}

BPE='--bpe sentencepiece --sentencepiece-model '${MODEL_DIR}'/bpe.model'
LOG_FILE=${MODEL_DIR}/log.out


########################################################
# 1. Tokenize data
########################################################
if [ ${tokenize} = "1" ]; then
# echo "================tokenizing training data================"
# python3 /home/sli136/l2mt/scripts/spm_tokenizer.py --train --encode \
#     --src-file ${RAW_SOURCE}/train.${rawfile_key}.${SRC} \
#     --tgt-file ${RAW_SOURCE}/train.${rawfile_key}.${TGT} \
#     --encode-files \
#     ${RAW_SOURCE}/train.${rawfile_key}.${SRC} \
#     ${RAW_SOURCE}/train.${rawfile_key}.${TGT} \
#     ${RAW_SOURCE}/dev.${rawfile_key}.${SRC} \
#     ${RAW_SOURCE}/dev.${rawfile_key}.${TGT} \
#     ${RAW_SOURCE}/test.${rawfile_key}.${SRC} \
#     ${RAW_SOURCE}/test.${rawfile_key}.${TGT} \
#     /home/sli136/l2mt/data/raw/test.l2.${SRC} \
#     /home/sli136/l2mt/data/raw/test.l2.${TGT} \
#     /home/sli136/l2mt/data/raw/test.ref.${SRC} \
#     /home/sli136/l2mt/data/raw/test.ref.${TGT} \
#     --encode-dest \
#     ${BPE_DIR}/train.${rawfile_key}.bpe.${SRC} \
#     ${BPE_DIR}/train.${rawfile_key}.bpe.${TGT} \
#     ${BPE_DIR}/dev.${rawfile_key}.bpe.${SRC} \
#     ${BPE_DIR}/dev.${rawfile_key}.bpe.${TGT} \
#     ${BPE_DIR}/test.${rawfile_key}.bpe.${SRC} \
#     ${BPE_DIR}/test.${rawfile_key}.bpe.${TGT} \
#     ${BPE_DIR}/test.l2.bpe.${SRC} \
#     ${BPE_DIR}/test.l2.bpe.${TGT} \
#     ${BPE_DIR}/test.ref.bpe.${SRC} \
#     ${BPE_DIR}/test.ref.bpe.${TGT} \
#     --model-dir ${MODEL_DIR}
# echo "================done tokenizing training data================"
echo "================tokenizing training data================"
python3 /home/sli136/l2mt/scripts/spm_tokenizer.py --encode \
    --src-file ${RAW_SOURCE}/train.${rawfile_key}.${SRC} \
    --tgt-file ${RAW_SOURCE}/train.${rawfile_key}.${TGT} \
    --encode-files \
    ${REF_L2_DIR}/test.l2.${SRC} \
    ${REF_L2_DIR}/test.l2.${TGT} \
    ${REF_L2_DIR}/test.ref.${SRC} \
    ${REF_L2_DIR}/test.ref.${TGT} \
    --encode-dest \
    ${BPE_DIR}/test.l2.bpe.${SRC} \
    ${BPE_DIR}/test.l2.bpe.${TGT} \
    ${BPE_DIR}/test.ref.bpe.${SRC} \
    ${BPE_DIR}/test.ref.bpe.${TGT} \
    --model-dir ${MODEL_DIR}
echo "================done tokenizing training data================"
fi

########################################################
# 2. Fairseq preprocess data
########################################################
if [ ${preprocess} = "1" ]; then
# echo "================preprocessing training data================"
# fairseq-preprocess --source-lang $SRC --target-lang $TGT \
#     --trainpref ${BPE_DIR}/train.${rawfile_key}.bpe \
#     --validpref ${BPE_DIR}/dev.${rawfile_key}.bpe \
#     --testpref ${BPE_DIR}/test.${rawfile_key}.bpe \
#     --bpe sentencepiece \
#     --destdir ${BASE_BIN} \
#     --joined-dictionary \
#     --workers 20

echo "================preprocessing ref eval data================"
echo "" > ${BPE_DIR}/train.ref.bpe.${SRC}
echo "" > ${BPE_DIR}/train.ref.bpe.${TGT}
echo "" > ${BPE_DIR}/dev.ref.bpe.${SRC}
echo "" > ${BPE_DIR}/dev.ref.bpe.${TGT}
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.ref.bpe \
    --validpref ${BPE_DIR}/dev.ref.bpe \
    --testpref ${BPE_DIR}/test.ref.bpe \
    --bpe sentencepiece \
    --destdir ${REF_BIN} \
    --tgtdict ${BASE_BIN}/dict.${TGT}.txt \
    --srcdict ${BASE_BIN}/dict.${SRC}.txt \
    --workers 20

echo "================preprocessing l2 eval data================"
echo "" > ${BPE_DIR}/train.l2.bpe.${SRC}
echo "" > ${BPE_DIR}/train.l2.bpe.${TGT}
echo "" > ${BPE_DIR}/dev.l2.bpe.${SRC}
echo "" > ${BPE_DIR}/dev.l2.bpe.${TGT}
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.l2.bpe \
    --validpref ${BPE_DIR}/dev.l2.bpe \
    --testpref ${BPE_DIR}/test.l2.bpe \
    --bpe sentencepiece \
    --destdir ${L2_BIN} \
    --tgtdict ${BASE_BIN}/dict.${TGT}.txt \
    --srcdict ${BASE_BIN}/dict.${SRC}.txt \
    --workers 20
echo "================done preprocessing data================"
fi

########################################################
# 1. Fairseq training
########################################################
if [ ${train} = "1" ]; then
MAX_UPDATES=400000
ARCH=transformer
FREQ=1
MAX_TOKENS=4096

LAYER=4
DIM=512
FFN_DIM=2048
HEADS=4

# Train (comment for evaluation)
echo "================STARTING TRAINING================"
WANDB__SERVICE_WAIT=300 fairseq-train ${BASE_BIN} --arch ${ARCH}  --task translation \
 ${BPE} \
    --encoder-layers ${LAYER} --encoder-ffn-embed-dim ${FFN_DIM} --encoder-embed-dim ${DIM} --encoder-attention-heads ${HEADS} --encoder-normalize-before \
    --decoder-layers ${LAYER} --decoder-ffn-embed-dim ${FFN_DIM} --decoder-embed-dim ${DIM} --decoder-attention-heads ${HEADS} --decoder-normalize-before \
    --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.3 --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 --max-tokens ${MAX_TOKENS} \
    --max-update ${MAX_UPDATES} --update-freq ${FREQ} \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0004 --stop-min-lr 1e-09 \
    --clip-norm 0.0 --weight-decay 0.0 --label-smoothing 0.1 \
    --criterion label_smoothed_cross_entropy \
    --save-interval-updates 5000 --keep-interval-updates 1 \
    --validate-interval-updates 5000 --patience 20 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --log-file ${LOG_FILE} --log-format simple --log-interval 500 \
    --save-dir ${MODEL_DIR} --wandb-project l2mt --seed 42 
echo "================done training================"
fi

########################################################
# 1. Fairseq evaluate
########################################################
if [ ${eval} = "1" ]; then
echo "================evaluating ref (corrected) data================"
RESULT_PATH=${MODEL_DIR}'/results/ref/'
mkdir -p ${RESULT_PATH}
WANDB__SERVICE_WAIT=300 fairseq-generate ${REF_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} \
    --spm ${MODEL_DIR}/bpe.model \
    --bertscore --bleu --ter --sacrebleu

echo "================evaluating l2 (crappy) data================"
RESULT_PATH=${MODEL_DIR}'/results/l2/'
mkdir -p ${RESULT_PATH}
WANDB__SERVICE_WAIT=300 fairseq-generate ${L2_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} \
    --spm ${MODEL_DIR}/bpe.model \
    --bertscore --bleu --ter --sacrebleu
fi