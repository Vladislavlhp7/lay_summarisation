#!/bin/bash

SEED_DIR=./data/input/rouge
MODEL="yikuan8/Clinical-Longformer"
SEED=42

python -m laysummarisation.model.train \
	--ftrain "${SEED_DIR}/${CORPUS}_train.jsonl" \
	--fvalid "${SEED_DIR}/${CORPUS}_val.jsonl" \
	--output_dir tmp \
	--seed ${SEED} \
	--model_checkpoint ${MODEL} \
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
