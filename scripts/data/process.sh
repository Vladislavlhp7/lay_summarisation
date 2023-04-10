#!/bin/bash
# Script to pre-process all data.

DATA_DIR=./data/task1_development/
SAVE_DIR=./data/input/

mkdir -p ${SAVE_DIR}

# Pre-process train
python -m laysummarisation.preprocess \
  --fname ${DATA_DIR}/train/${CORPUS}_train.jsonl \
  --output ${SAVE_DIR}/${CORPUS}_train.jsonl \
  --lex_sent 25 

# Pre-process val
python -m laysummarisation.preprocess \
  --fname ${DATA_DIR}/val/${CORPUS}_val.jsonl \
  --output ${SAVE_DIR}/${CORPUS}_val.jsonl \
  --lex_sent 25 
