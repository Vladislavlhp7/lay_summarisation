#!/bin/bash

SEED_DIR=./data/input/rouge
MODEL="yikuan8/Clinical-Longformer"
CORPUS="eLife"
SEED=42

python -m laysummarisation.model.clinical_longformer \
	--ftrain "${SEED_DIR}/${CORPUS}_train.jsonl" \
	--fvalid "${SEED_DIR}/${CORPUS}_val.jsonl" \
	--save_dir tmp \
	--seed ${SEED} \
	--model_checkpoint ${MODEL} \
	--device "gpu" \
	--max_encode 1028 \
	--max_decode 1024 \
	--min_encode 512 \
	--attention_window 32 \
	--nbeams 4 \
	--length_penalty 2.0 \
	--lr 0.00003 \
	--batch_size 1 \
	--epochs 5 \
	--save_steps 1000 \
	--eval_steps 1000 \
	--weight_decay 0.01 \
	--metric "rouge2_f" \
	--logging_steps 1000
