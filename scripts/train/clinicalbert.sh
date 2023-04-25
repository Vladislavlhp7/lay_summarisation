#!/bin/bash --login
#$ -N clinical_bert
#$ -cwd

#$ -l v100
#$ -pe smp.pe 8

module load libs/cuda
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load apps/binapps/anaconda3/2021.11  # Python 3.9.7
module load libs/nvidia-hpc-sdk/23.1
module load compilers/gcc/9.3.0

SEED_DIR=./data/tmp/extractive/rouge

python -m laysummarisation.model.extractor_model \
  --fname ${SEED_DIR}/${CORPUS}_train.csv \
  --lr 5e-5