#!/bin/bash

SEED_DIR=./data/input/rouge
MODEL="yikuan8/Clinical-Longformer"
CORPUS="eLife"
CHECKPOINT="tmp/${CORPUS}/checkpoint-5430"
SEED=42

python -m laysummarisation.model.clinical_longformer \
	--ftrain "${SEED_DIR}/${CORPUS}_train.jsonl" \
	--fvalid "${SEED_DIR}/${CORPUS}_val.jsonl" \
	--corpus ${CORPUS} \
	--save_dir tmp/${CORPUS} \
	--seed ${SEED} \
	--model ${MODEL} \
	--checkpoint ${CHECKPOINT} \
	--device "cuda" \
	--max_encode 1024 \
	--max_decode 512 \
	--min_encode 512 \
	--attention_window 32 \
	--nbeams 4 \
	--length_penalty 1.0 \
	--batch_size 1 \
	--epochs 10 \
	--weight_decay 0.01 \
	--warmup_steps 1000 \
	--gradient_accum_steps 4 \
	--metric "rouge2_f" \
	--save_strategy "no" \
	--logging_strategy "steps" \
	--eval_strategy "epoch" \
	--save_steps 2000 \
	--eval_steps 2000 \
	--logging_steps 1000 \
	--load_best_model_at_end \
	--fp16 \
	--fp16_full_eval
