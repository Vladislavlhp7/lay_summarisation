#!/bin/bash

# Process the data using rouge maximisation.
SEED_DIR=./data/input/rouge
MODEL=./weights/Clinical-Longformer
SEED=42

mkdir -p ${SEED_DIR}

python -m laysummarisation.process.greedy_rouge \
	--fname ./data/orig/train/${CORPUS}_train.jsonl \
	--output ${SEED_DIR}/${CORPUS}_train.jsonl \
	--nsent 10 \
	--mode "split" \
	--workers 6 \
	--nrows 10

# set nwors to 0 for the whole corpus
