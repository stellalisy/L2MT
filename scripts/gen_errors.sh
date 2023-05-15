#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt/scripts
#$ -V
#$ -N zh-both-train
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=4G,hostname=octopod|c*|b*
#$ -t 1
#$ -o /home/sli136/l2mt/output/cpu-shell/

conda activate l2mt
cd /home/sli136/l2mt/scripts/

splits=${1}
run_name=${2}  # typo, runon, both, para

if [ "${run_name}" = "typo" ]; then
    errors="typo"
    probs="0.15"
elif [ "${run_name}" = "runon" ]; then
    errors="runon_line-runon_comb"
    probs="0.9-0.5"
elif [ "${run_name}" = "both" ]; then
    errors="typo-runon_line-runon_comb"
    probs="0.15-0.9-0.4"
elif [ "${run_name}" = "para" ]; then
    errors="paraphrase"
    probs="0.5"
fi

for split in ${splits//,/ }; do
echo "Generating ${errors} errors for ${split} with probability ${probs}"
python3 /home/sli136/l2mt/scripts/gen_errors.py \
    -s /home/sli136/l2mt/data/raw/zh-clean/${split}.zh-clean.en \
    -t /home/sli136/l2mt/data/nmt-grammar-noise/zh-artl2-errors/${split}.${run_name} \
    -l /home/sli136/l2mt/output/cpu-shell/mylog/ \
    -ce \
    -e ${errors//-/ } \
    -p ${probs//-/ } \
    --tgt-lang zh
done



# echo "Generating ${errors} errors for train set ${train_set_i} with probability ${probs}"
# python3 /home/sli136/l2mt/scripts/gen_errors.py \
#     -s /home/sli136/l2mt/data/raw/train_split/train.clean.${train_set_i} \
#     -t /home/sli136/l2mt/data/nmt-grammar-noise/en-artl2-errors/train-para-split/${split}.${run_name}.${train_set_i} \
#     -l /home/sli136/l2mt/output/cpu-shell/mylog/paraphrase-train/ \
#     -ce \
#     -e ${errors//-/ } \
#     -p ${probs//-/ }
