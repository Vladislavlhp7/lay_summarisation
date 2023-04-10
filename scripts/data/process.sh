#!/bin/bash
# Script to pre-process all data.

DATA_DIR=./data/orig/
SAVE_DIR=./data/input/

mkdir -p ${SAVE_DIR}

# Pre-process train
python -m laysummarisation.preprocess \
  --fname ${DATA_DIR}/train/${CORPUS}_train.jsonl \
  --output ${SAVE_DIR}/${CORPUS}_train.jsonl \
  --lex_sent 25 \
  # --entries 1

echo "------ Finished pre-processing train of ${CORPUS} ------"

# Pre-process val
python -m laysummarisation.preprocess \
  --fname ${DATA_DIR}/val/${CORPUS}_val.jsonl \
  --output ${SAVE_DIR}/${CORPUS}_val.jsonl \
  --lex_sent 25 \
  # --entries 1

echo "------ Finished pre-processing val of ${CORPUS} ------"
