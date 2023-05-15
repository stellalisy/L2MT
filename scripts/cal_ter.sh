#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -N para-grammar-eval
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

exp_model_dir=/export/c11/sli136/l2mt/en-es/models/para-grammar
echo "================evaluating ref================"
python scripts/cal_ter.py \
    -d ${exp_model_dir}/results/ref \
    --spm ${exp_model_dir}/bpe.model \
    --bertscore --bleu --ter --sacrebleu

echo "================evaluating l2================"
python scripts/cal_ter.py \
    -d ${exp_model_dir}/results/l2 \
    --spm ${exp_model_dir}/bpe.model \
    --bertscore --bleu --ter --sacrebleu

# folder=/export/c11/sli136/l2mt/models
# keys=clean_4,mixed_4,artl2-typo,artl2-runon,artl2-both,all-mixed
# eval_sets=ref,l2

# for key in ${keys//,/ }; do
#     for eval_set in ${eval_sets//,/ }; do
#         echo "================evaluating ${key} ${eval_set}================"
#         python scripts/cal_ter.py \
#             -d ${folder}/${key}/results/${eval_set} \
#             --spm ${folder}/${key}/bpe.model \
#             --meteor
#             # --bertscore --meteor --bleu --ter --sacrebleu --nltk
#     done
# done

# plms=m2m100,mbart

# for plm in ${plms//,/ }; do
#     for eval_set in ${eval_sets//,/ }; do
#         echo "================evaluating ${plm} ${eval_set}================"
#         python scripts/cal_ter.py \
#             -p /export/c11/sli136/l2mt/models/plm/test.${eval_set}.pred.${plm} \
#             -t /home/sli136/l2mt/data/raw/test.${eval_set}.es \
#             --spm ${folder}/clean_4/bpe.model \
#             --meteor
#             # --bertscore --meteor --bleu --ter --sacrebleu --nltk
#     done
# done

