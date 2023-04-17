#!/bin/bash
#$ -cwd

#$ -pe smp.pe 8

export OMP_NUM_THREADS=$NSLOTS

module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load tools/env/proxy2

source activate nlu

./scripts/process_rouge.sh

