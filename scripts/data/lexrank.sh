#!/bin/bash
# Script to pre-process all data.

SAVE_DIR=./data/input/lexrank
ORIG_DIR=./data/orig/train

mkdir -p ${SAVE_DIR}

python -m laysummarisation.process.lexrank \
	--data_dir "${ORIG_DIR}" \
	--output_dir "${SAVE_DIR}" \
	--corpus "${CORPUS}" \
	--nrows 10 \
	--nsent 25 \
	--seed 42 \
	--workers 6
# --all \

echo "------ Finished pre-processing train of ${CORPUS} ------"

# Pre-process val
# python -m laysummarisation.process.lexrank \
#   --fname ${DATA_DIR}/val/${CORPUS}_val.jsonl \
#   --output ${SAVE_DIR}/${CORPUS}_val.jsonl \
#   --lex_sent 25 \
#   --nrows 2

# echo "------ Finished pre-processing val of ${CORPUS} ------"
