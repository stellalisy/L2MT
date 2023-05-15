#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N mixed-reproduce
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
KEY=mixed-reproduce
rawfile_key='mixed'                                        # name identifier for raw data in the format "train.${rawfile_key}.en"
RAW_SOURCE='/home/sli136/l2mt/data/raw/'${rawfile_key}     # raw (untokenized) data (artificial mixed data directory, doesn't include l2 and ref data)
DEST_DIR='/export/c11/sli136/l2mt/data/input_data_'${KEY}  # parent directory for all tokenized data to store input for training
BPE_DIR=${DEST_DIR}'/bpe_tokenized'                        # bpe tokenized data for fairseq training
BASE_BIN=${DEST_DIR}'/base_bin'                            # fairseq binarized data for fairseq training
L2_BIN=${DEST_DIR}'/l2_bin'                                # fairseq binarized data for l2 eval (original crappy sentences)
REF_BIN=${DEST_DIR}'/ref_bin'                              # fairseq binarized data for ref eval (human corrected sentences)
MODEL_DIR='/export/c11/sli136/l2mt/models/'${KEY}          # parent directory for all models to store output from training

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
# 1. Tokenize data
########################################################
if [ ${tokenize} = "1" ]; then
echo "================tokenizing training data================"
python3 /home/sli136/l2mt/scripts/spm_tokenizer.py --train --encode \
    --src-file ${RAW_SOURCE}/train.${rawfile_key}.en \
    --tgt-file ${RAW_SOURCE}/train.${rawfile_key}.es \
    --encode-files \
    ${RAW_SOURCE}/train.${rawfile_key}.en \
    ${RAW_SOURCE}/train.${rawfile_key}.es \
    ${RAW_SOURCE}/dev.${rawfile_key}.en \
    ${RAW_SOURCE}/dev.${rawfile_key}.es \
    ${RAW_SOURCE}/test.${rawfile_key}.en \
    ${RAW_SOURCE}/test.${rawfile_key}.es \
    /home/sli136/l2mt/data/raw/test.l2.en \
    /home/sli136/l2mt/data/raw/test.l2.es \
    /home/sli136/l2mt/data/raw/test.ref.en \
    /home/sli136/l2mt/data/raw/test.ref.es \
    --encode-dest \
    ${BPE_DIR}/train.${rawfile_key}.bpe.en \
    ${BPE_DIR}/train.${rawfile_key}.bpe.es \
    ${BPE_DIR}/dev.${rawfile_key}.bpe.en \
    ${BPE_DIR}/dev.${rawfile_key}.bpe.es \
    ${BPE_DIR}/test.${rawfile_key}.bpe.en \
    ${BPE_DIR}/test.${rawfile_key}.bpe.es \
    ${BPE_DIR}/test.l2.bpe.en \
    ${BPE_DIR}/test.l2.bpe.es \
    ${BPE_DIR}/test.ref.bpe.en \
    ${BPE_DIR}/test.ref.bpe.es \
    --model-dir ${MODEL_DIR}
echo "================done tokenizing training data================"
fi

########################################################
# 2. Fairseq preprocess data
########################################################
if [ ${preprocess} = "1" ]; then
echo "================preprocessing training data================"
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.${rawfile_key}.bpe \
    --validpref ${BPE_DIR}/dev.${rawfile_key}.bpe \
    --testpref ${BPE_DIR}/test.${rawfile_key}.bpe \
    --bpe sentencepiece \
    --destdir ${BASE_BIN} \
    --joined-dictionary \
    --workers 20

echo "================preprocessing ref eval data================"
echo "" > ${BPE_DIR}/train.ref.bpe.en
echo "" > ${BPE_DIR}/train.ref.bpe.es
echo "" > ${BPE_DIR}/dev.ref.bpe.en
echo "" > ${BPE_DIR}/dev.ref.bpe.es
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.ref.bpe \
    --validpref ${BPE_DIR}/dev.ref.bpe \
    --testpref ${BPE_DIR}/test.ref.bpe \
    --bpe sentencepiece \
    --destdir ${REF_BIN} \
    --tgtdict ${BASE_BIN}/dict.es.txt \
    --srcdict ${BASE_BIN}/dict.en.txt \
    --workers 20

echo "================preprocessing l2 eval data================"
echo "" > ${BPE_DIR}/train.l2.bpe.en
echo "" > ${BPE_DIR}/train.l2.bpe.es
echo "" > ${BPE_DIR}/dev.l2.bpe.en
echo "" > ${BPE_DIR}/dev.l2.bpe.es
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.l2.bpe \
    --validpref ${BPE_DIR}/dev.l2.bpe \
    --testpref ${BPE_DIR}/test.l2.bpe \
    --bpe sentencepiece \
    --destdir ${L2_BIN} \
    --tgtdict ${BASE_BIN}/dict.es.txt \
    --srcdict ${BASE_BIN}/dict.en.txt \
    --workers 20
echo "================done preprocessing data================"
fi

########################################################
# 1. Fairseq training
########################################################
if [ ${train} = "1" ]; then
MAX_UPDATES=200000
ARCH=transformer
FREQ=1
MAX_TOKENS=4096

LAYER=4
DIM=512
FFN_DIM=2048
HEADS=4

DROPOUT=0.2

# Train (comment for evaluation)
echo "================STARTING TRAINING================"
fairseq-train ${BASE_BIN} --arch ${ARCH}  --task translation \
 ${BPE} \
    --encoder-layers ${LAYER} --encoder-embed-dim ${DIM} --encoder-ffn-embed-dim ${FFN_DIM} --encoder-attention-heads ${HEADS} --encoder-normalize-before \
    --decoder-layers ${LAYER} --decoder-embed-dim ${DIM} --decoder-ffn-embed-dim ${FFN_DIM} --decoder-attention-heads ${HEADS} --decoder-normalize-before \
    --attention-dropout 0.1 --relu-dropout 0.0 --dropout ${DROPOUT} --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 --max-tokens ${MAX_TOKENS} \
    --max-update ${MAX_UPDATES} --update-freq ${FREQ} \
    --optimizer adam --adam-eps 1e-09 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 24000 --lr 0.0004 --stop-min-lr 1e-09 \
    --clip-norm 0.0 --weight-decay 0.0 --label-smoothing 0.1 \
    --criterion label_smoothed_cross_entropy \
    --save-interval-updates 5000 --keep-interval-updates 1 \
    --validate-interval-updates 5000 --patience 20 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe \
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
fairseq-generate ${REF_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} \
    --spm ${MODEL_DIR}/bpe.model \
    --bertscore --meteor --bleu --ter --sacrebleu --nltk

echo "================evaluating l2 (crappy) data================"
RESULT_PATH=${MODEL_DIR}'/results/l2/'
mkdir -p ${RESULT_PATH}
fairseq-generate ${L2_BIN} --path $MODEL_DIR/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH} \
    --spm ${MODEL_DIR}/bpe.model \
    --bertscore --meteor --bleu --ter --sacrebleu --nltk
fi