#!/bin/bash

# Process the data using rouge maximisation.
SEED_DIR=./data/tmp/extractive/${ALGO}
INPUT_DIR=./data/input/${ALGO}
ORIG_DIR=./data/orig/train
mkdir -p "${SEED_DIR}"

python -m laysummarisation.process.binary_sentences \
	--data_dir "${INPUT_DIR}" \
	--orig_dir "${ORIG_DIR}" \
	--output "${SEED_DIR}" \
	--corpus "${CORPUS}" \
	--narticles 10 \
	--seed 42 \
	--all \
	--balance
