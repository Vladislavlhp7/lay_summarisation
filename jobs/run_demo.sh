#!/bin/bash --login
#$ -cwd

#|$ -l v100
#|$ -pe smp.pe 8

module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load tools/env/proxy2

# create venv
source activate nlu

./scripts/run_demo.sh
