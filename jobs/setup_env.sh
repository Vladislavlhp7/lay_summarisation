#!/bin/bash --login
#$ -cwd

module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load tools/env/proxy2

# create venv
conda create -y -n nlu python=3.9
source activate nlu

chmod +x ./scripts/**/*.sh

./scripts/install.sh
