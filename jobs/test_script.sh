#!/bin/bash
#$ -cwd

#$ -l v100-2
#$ -pe smp.pe 8

export OMP_NUM_THREADS=$NSLOTS

# load modules
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load tools/env/proxy2

# source activate 
source activate nlu

# ensure packages are up to date
pip install -r requirements.txt
