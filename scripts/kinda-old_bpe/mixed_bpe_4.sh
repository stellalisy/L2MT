#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N mixed-4-eval
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

KEY=mixed_4
DATA_DIR='/home/sli136/l2mt/data/bpe/input_data_'${KEY}
BPE_DIR=${DATA_DIR}'/bpe_tokenized'
DATA_BIN=${DATA_DIR}'/base_bin'
REF_BIN=${DATA_DIR}'/ref_bin'
L2_BIN=${DATA_DIR}'/l2_bin'

SAVE_PATH='/export/c11/sli136/l2mt/models/'${KEY}
[ ! -d ${SAVE_PATH} ] && mkdir -p ${SAVE_PATH}

SRC='en'
TGT='es'

BPE='--bpe sentencepiece --sentencepiece-model '${SAVE_PATH}'/bpe.model'

echo "================preprocessing training data================"
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref ${BPE_DIR}/train.mixed.bpe \
    --validpref ${BPE_DIR}/dev.mixed.bpe \
    --testpref ${BPE_DIR}/test.mixed.bpe \
    --bpe sentencepiece \
    --destdir ${DATA_BIN} \
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
    --tgtdict ${DATA_BIN}/dict.es.txt \
    --srcdict ${DATA_BIN}/dict.en.txt \
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
    --tgtdict ${DATA_BIN}/dict.es.txt \
    --srcdict ${DATA_BIN}/dict.en.txt \
    --workers 20

echo "================done preprocessing data================"

MAX_UPDATES=200000
ARCH=transformer
FREQ=1
MAX_TOKENS=8192

LAYER=4
DIM=512
FFN_DIM=2048
HEADS=4

LOG_FILE=${SAVE_PATH}/log.out

# Train (comment for evaluation)
echo "================STARTING TRAINING================"
fairseq-train ${DATA_BIN} --arch ${ARCH}  --task translation \
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
    --save-dir ${SAVE_PATH} --wandb-project l2mt --seed 42 
echo "================done training================"

# Evaluate
echo "================evaluating ref (corrected) data================"
RESULT_PATH=${SAVE_PATH}'/results/ref/'
mkdir -p ${RESULT_PATH}
fairseq-generate ${REF_BIN} --path $SAVE_PATH/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH}

echo "================evaluating l2 (crappy) data================"
RESULT_PATH=${SAVE_PATH}'/results/l2/'
mkdir -p ${RESULT_PATH}
fairseq-generate ${L2_BIN} --path $SAVE_PATH/checkpoint_best.pt \
    ${BPE} \
    --task translation \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --sacrebleu \
    --source-lang ${SRC} --target-lang ${TGT} \
    --results-path ${RESULT_PATH}
python /home/sli136/l2mt/scripts/cal_ter.py -d ${RESULT_PATH}