#!/bin/bash --login
#$ -N clinical_longformer_
#$ -cwd

#$ -l v100
#$ -pe smp.pe 8

module load apps/binapps/pytorch/1.11.0-39-gpu-cu113

SEED_DIR=./log/multitask_am/st/${CORPUS}/${SEED}

python -m laysummarisation.model.train \
  --ftrain ${SEED_DIR}/data/train.mrp \
  --fvalid ${SEED_DIR}/data/dev.mrp \
  --output_dir tmp \
  --seed ${SEED} \
  --model_checkpoint ./weights/Clinical-Longformer \
  --device "gpu" \
  --max_encode 4096 \
  --max_decode 1024 \
  --lr 0.00003 \
  --batch_size 4 \
  --epochs 20 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --weight_decay 0.01 \
  --metric "rouge2_f"


