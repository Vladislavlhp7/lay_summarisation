#!/bin/bash

# Process the data using rouge maximisation.
SEED_DIR=./data/tmp/extractive/${ALGO}
mkdir -p ${SEED_DIR}

python -m laysummarisation.process.greedy_rouge \
    --summary_fname ./data/${ALGO}/${CORPUS}_train.jsonl \
    --fname ./data/orig/train/${CORPUS}_train.jsonl \
    --output ${SEED_DIR}/${CORPUS}_train.jsonl \
    --narticles 10000000
