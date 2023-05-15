#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N mixed-4-tokenize-all
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=octopod|c*
#$ -q g.q
#$ -t 1
#$ -j y -o /home/sli136/l2mt/output/tokenize/

source /home/sli136/scripts/acquire_gpu

conda activate l2mt
cd /home/sli136/l2mt

KEY=mixed_4
RAW_SOURCE='/home/sli136/l2mt/data/raw'
DATA_DIR='/home/sli136/l2mt/data/bpe/input_data_mixed_4/bpe_tokenized'
BASE_BIN='/home/sli136/l2mt/data/bpe/input_data_mixed_4/base_bin'
L2_BIN='/home/sli136/l2mt/data/bpe/input_data_mixed_4/l2_bin'
REF_BIN='/home/sli136/l2mt/data/bpe/input_data_mixed_4/ref_bin'
MODEL_DIR='/export/c11/sli136/l2mt/'${KEY}

[ ! -d ${BASE_BIN} ] && mkdir -p ${BASE_BIN}
[ ! -d ${REF_BIN} ] && mkdir -p ${REF_BIN}
[ ! -d ${L2_BIN} ] && mkdir -p ${L2_BIN}
[ ! -d ${MODEL_DIR} ] && mkdir -p ${MODEL_DIR}

python3 /home/sli136/l2mt/spm_tokenizer.py \
    --src-file ${RAW_SOURCE}/train.mixed.en \
    --tgt-file ${RAW_SOURCE}/train.mixed.es \
    --encode-files \
    ${RAW_SOURCE}/train.mixed.en \
    ${RAW_SOURCE}/train.mixed.es \
    ${RAW_SOURCE}/dev.mixed.en \
    ${RAW_SOURCE}/dev.mixed.es \
    ${RAW_SOURCE}/test.mixed.en \
    ${RAW_SOURCE}/test.mixed.es \
    ${RAW_SOURCE}/test.l2.en \
    ${RAW_SOURCE}/test.l2.es \
    ${RAW_SOURCE}/test.ref.en \
    ${RAW_SOURCE}/test.ref.es \
    --encode-dest \
    ${DATA_DIR}/train.mixed.bpe.en \
    ${DATA_DIR}/train.mixed.bpe.es \
    ${DATA_DIR}/dev.mixed.bpe.en \
    ${DATA_DIR}/dev.mixed.bpe.es \
    ${DATA_DIR}/test.mixed.bpe.en \
    ${DATA_DIR}/test.mixed.bpe.es \
    ${DATA_DIR}/test.l2.bpe.en \
    ${DATA_DIR}/test.l2.bpe.es \
    ${DATA_DIR}/test.ref.bpe.en \
    ${DATA_DIR}/test.ref.bpe.es \
    --model-dir ${MODEL_DIR} --train --encode
