#!/bin/bash --login
#$ -N clinical_longformer_
#$ -cwd

#$ -l v100
#$ -pe smp.pe 8

module load libs/cuda
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load apps/binapps/anaconda3/2021.11  # Python 3.9.7

python3 ../laysummarisation/model/train.py
