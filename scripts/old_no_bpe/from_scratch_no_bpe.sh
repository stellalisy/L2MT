#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -V
#$ -N l2-generate
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=octopod
#$ -q g.q
#$ -t 1
#$ -o /home/sli136/l2mt/output/transformer

# b1[123456789]|c0*|c1[123456789]

source /home/sli136/scripts/acquire_gpu

# Activate dev environments and call programs
conda activate l2mt
cd /home/sli136/l2mt

DATA_DIR='/home/sli136/l2mt/input_data'
DATA_BIN=${DATA_DIR}'/data_bin'
src='en'
tgt='es'

# fairseq-preprocess --source-lang $src --target-lang $tgt \
#     --trainpref /home/sli136/l2mt/nmt-grammar-noise/en-es-experiments/train.clean \
#     --validpref /home/sli136/l2mt/input_data/retrieved_ref/valid.ref \
#     --testpref /home/sli136/l2mt/input_data/retrieved_ref/test.ref \
#     --joined-dictionary \
#     --destdir ${DATA_BIN} \
#     --workers 20

# echo "done preprocessing data"

MAX_UPDATES=1000000
ARCH=transformer
FREQ=1
MAX_TOKENS=200

LAYER=6
DIM=512
FFN_DIM=1024
HEADS=8
SAVE_PATH='/export/c11/sli136/l2mt/checkpoints'

mkdir -p ${SAVE_PATH}
LOG_FILE=${SAVE_PATH}/log.out
DATA_BIN=${DATA_DIR}'/data_bin'
# Train (comment for evaluation)
fairseq-train ${DATA_BIN} --arch ${ARCH}  --task translation \
 --encoder-layers ${LAYER} --decoder-layers ${LAYER} --encoder-ffn-embed-dim ${FFN_DIM} --decoder-ffn-embed-dim ${FFN_DIM} \
 --encoder-embed-dim ${DIM} --decoder-embed-dim ${DIM} --encoder-attention-heads ${HEADS} --decoder-attention-heads ${HEADS} --attention-dropout 0.1 --relu-dropout 0.0 \
 --decoder-normalize-before --encoder-normalize-before --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
 --max-update ${MAX_UPDATES} --update-freq ${FREQ}  --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0004 --stop-min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 --best-checkpoint-metric loss --max-tokens ${MAX_TOKENS}  --validate-interval-updates 500 --save-interval-updates 500 --save-interval 2 \
 --keep-interval-updates 1  --validate-interval 1000  --seed 42 --log-format simple --log-interval 100 \
 --optimizer adam --log-file ${LOG_FILE}\
 --save-dir ${SAVE_PATH}  --skip-invalid-size-inputs-valid-test --wandb-project l2mt
exit

# echo "done training"
# Evaluate
# mkdir -p ${SAVE_PATH}/results

# DATA_BIN=${DATA_DIR}'/ref_bin'

mkdir -p ${SAVE_PATH}/results/ref/
DATA_BIN=${DATA_DIR}'/ref_bin'
fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
    --task translation \
    --remove-bpe \
    --sacrebleu \
    --no-progress-bar \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --scoring bleu \
    --source-lang ${src} --target-lang ${tgt} \
    --results-path ${SAVE_PATH}/results/ref/


mkdir -p ${SAVE_PATH}/results/l2/
DATA_BIN=${DATA_DIR}'/l2_bin'
fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
    --task translation \
    --remove-bpe \
    --sacrebleu \
    --no-progress-bar \
    --beam 5 --lenpen 1.0 \
    --batch-size 16  \
    --scoring bleu \
    --source-lang ${src} --target-lang ${tgt} \
    --results-path ${SAVE_PATH}/results/l2/