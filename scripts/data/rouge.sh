#!/bin/bash

# Process the data using rouge maximisation.
SAVE_DIR=./data/input/lexrank
ORIG_DIR=./data/orig/train

mkdir -p ${SAVE_DIR}

python -m laysummarisation.process.greedy_rouge \
	--data_dir "${ORIG_DIR}" \
	--output_dir "${SAVE_DIR}" \
	--corpus "${CORPUS}" \
	--nrows 0 \
	--nsent 25 \
	--seed 42 \
	--workers 6
# --all \

# python -m laysummarisation.process.greedy_rouge \
# 	--fname ./data/orig/train/${CORPUS}_train.jsonl \
# 	--output ${SEED_DIR}/${CORPUS}_train.jsonl \
# 	--nsent 10 \
# 	--mode "split" \
# 	--workers 8 \
# 	--nrows 0

# set nwors to 0 for the whole corpus
