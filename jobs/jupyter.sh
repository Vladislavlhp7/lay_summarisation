module load apps/binapps/jupyter-notebook/5.5.0
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load tools/env/proxy2

source activate nlu

jupyter-notebook-csf -p 8
