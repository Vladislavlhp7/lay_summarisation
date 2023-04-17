#!/bin/bash

chmod +x scripts/data/*.sh

export CORPUS=eLife
./scripts/data/lexrank.sh

export CORPUS=PLOS
./scripts/data/lexrank.sh
