#!/bin/bash

# Process the data using rouge maximisation.
SEED_DIR=./data/input/rouge
MODEL=./weights/Clinical-Longformer
SEED=42

mkdir -p ${SEED_DIR}

python -m laysummarisation.process.binary_sentences \
	--data_dir "${INPUT_DIR}" \
	--orig_dir "${ORIG_DIR}" \
	--output "${SEED_DIR}" \
	--corpus "${CORPUS}" \
	--narticles 0 \
	--seed 42 \
	--all \
	--balance

# python -m laysummarisation.process.greedy_rouge \
# 	--fname ./data/orig/train/${CORPUS}_train.jsonl \
# 	--output ${SEED_DIR}/${CORPUS}_train.jsonl \
# 	--nsent 10 \
# 	--mode "split" \
# 	--workers 8 \
# 	--nrows 0

# set nwors to 0 for the whole corpus
