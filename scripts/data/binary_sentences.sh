#!/bin/bash

# Process the data using rouge maximisation.
SEED_DIR=./data/tmp/extractive/${ALGO}
mkdir -p ${SEED_DIR}

python -m laysummarisation.process.binary_sentences \
    --summary_fname ./data/input/${ALGO}/${CORPUS}_train.jsonl \
    --article_fname ./data/orig/train/${CORPUS}_train.jsonl \
    --output ${SEED_DIR}/${CORPUS}_train.csv \
    --narticles 2
