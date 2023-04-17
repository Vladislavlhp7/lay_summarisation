#!/bin/bash --login
#$ -N clinical_longformer_
#$ -cwd

#$ -l v100
#$ -pe smp.pe 8

module load libs/cuda
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load apps/binapps/anaconda3/2021.11  # Python 3.9.7
module load libs/nvidia-hpc-sdk/23.1
module load compilers/gcc/9.3.0

export OMP_NUM_THREADS=$NSLOTS

! pip install datasets

python3 ../laysummarisation/model/clinical_longformer.py