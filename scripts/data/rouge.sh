#!/bin/bash

# Process the data using rouge maximisation.
SAVE_DIR=./data/input/rouge
ORIG_DIR=./data/orig/val

mkdir -p ${SAVE_DIR}

python -m laysummarisation.process.greedy_rouge \
	--data_dir "${ORIG_DIR}" \
	--output_dir "${SAVE_DIR}" \
	--corpus "${CORPUS}" \
	--nrows 0 \
	--nsent 10 \
	--seed 42 \
	--workers 6
# --all \
