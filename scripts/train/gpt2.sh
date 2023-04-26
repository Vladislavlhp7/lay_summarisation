#!/bin/bash

SEED_DIR=./data/input/rouge
MODEL="gpt2"
CORPUS="eLife"
SEED=42

python -m laysummarisation.model.gpt2 \
	--ftrain "${SEED_DIR}/${CORPUS}_train.jsonl" \
	--fvalid "${SEED_DIR}/${CORPUS}_val.jsonl" \
	--corpus ${CORPUS} \
	--save_dir tmp/gpt2 \
	--seed ${SEED} \
	--model_checkpoint ${MODEL} \
	--device "cuda" \
	--temperature 0.7 \
	--max_encode 1024 \
	--lr 1e-5 \
	--batch_size 1 \
	--epochs 3 \
	--save_steps 2000
