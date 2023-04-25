#!/bin/bash --login
#$ -N clinical_longformer_
#$ -cwd

#$ -l v100
#$ -pe smp.pe 8

module load libs/cuda
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
module load apps/binapps/anaconda3/2021.11 # Python 3.9.7
module load libs/nvidia-hpc-sdk/23.1
module load compilers/gcc/9.3.0

SEED_DIR=./data/input
MODEL=./weights/Clinical-Longformer
SEED=42

python -m laysummarisation.model.train \
	--ftrain ${SEED_DIR}/${CORPUS}_train.jsonl \
	--fvalid ${SEED_DIR}/${CORPUS}_val.jsonl \
	--save_dir tmp \
	--seed ${SEED} \
	--model_checkpoint ${MODEL} \
	--device "gpu" \
	--atention_window 512 \
	--max_encode 4096 \
	--min_encode 128 \
	--max_decode 1024 \
	--nbeams 4 \
	--length_penalty 0.8 \
	--lr 0.00003 \
	--batch_size 4 \
	--epochs 20 \
	--save_steps 1000 \
	--eval_steps 1000 \
	--weight_decay 0.01 \
	--logging_steps 100 \
	--warmup_steps 1000 \
	--gradient_accum_steps 4 \
	--metric "rouge2_f"
# --early_stopping \
