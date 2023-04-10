#!/bin/bash
# This script is used to train the Clinical Longformer model on the eLife dataset
# Make sure data is initialised

export CORPUS=elife
./scripts/train/tune.sh
