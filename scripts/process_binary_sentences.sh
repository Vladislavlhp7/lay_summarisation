#!/bin/bash

chmod +x scripts/data/*.sh

# export CORPUS=eLife
# export ALGO=rouge
# ./scripts/data/binary_sentences.sh
#
export CORPUS="all"
export ALGO=rouge
./scripts/data/binary_sentences.sh
