#!/bin/bash

chmod +x scripts/data/*.sh

export CORPUS=eLife
export ALGO=rouge
./scripts/data/binary_sentences.sh

#export CORPUS=PLOS
#export ALGO=rouge
#./scripts/data/binary_sentences.sh
