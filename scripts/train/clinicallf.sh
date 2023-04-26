#!/bin/bash

SEED_DIR=./data/input/rouge
MODEL="yikuan8/Clinical-Longformer"
CORPUS="eLife"
SEED=42

python -m laysummarisation.model.clinical_longformer \
	--ftrain "${SEED_DIR}/${CORPUS}_train.jsonl" \
	--fvalid "${SEED_DIR}/${CORPUS}_val.jsonl" \
	--corpus ${CORPUS} \
	--save_dir tmp \
	--seed ${SEED} \
	--model_checkpoint ${MODEL} \
	--device "cuda" \
	--max_encode 1024 \
	--max_decode 512 \
	--min_encode 512 \
	--attention_window 32 \
	--nbeams 4 \
	--length_penalty 1.0 \
	--batch_size 1 \
	--epochs 5 \
	--save_steps 2000 \
	--eval_steps 2000 \
	--weight_decay 0.01 \
	--warmup_steps 1000 \
	--gradient_accum_steps 4 \
	--metric "rouge2_f" \
	--logging_steps 1000
