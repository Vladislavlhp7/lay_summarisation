#!/bin/bash
# This script is used to train the Clinical Longformer model on the PLOS dataset
# Make sure data is initialised

export CORPUS=plos
./scripts/train/tune.sh
