#!/usr/bin/env bash
#$ -wd /home/sli136/l2mt
#$ -V
#$ -N combine-grammar
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M sli136@jhu.edu
#$ -m e
#$ -l ram_free=10G,hostname=octopod|c*
#$ -q all.q
#$ -t 1
#$ -o /home/sli136/l2mt/output/cpu-shell/

conda activate l2mt
cd /home/sli136/l2mt

# python /home/sli136/l2mt/scripts/move_data.py -r 6 \
#     --output_dir /home/sli136/l2mt/data/raw/all-mixed-lrg \
#     --key all-mixed-lrg

out_dir=/export/c11/sli136/l2mt/en-es/data/raw/grammar-mixed
mkdir -p ${out_dir}

python /home/sli136/l2mt/scripts/move_data.py -r 2 \
    --output_dir ${out_dir} \
    --key grammar-mixed \
    --errors article nounnum prep sva

