#!/bin/bash

chmod +x scripts/data/*.sh

export CORPUS=eLife
./scripts/data/rouge.sh
#
export CORPUS=PLOS
./scripts/data/rouge.sh
