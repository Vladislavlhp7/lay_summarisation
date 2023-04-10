#!/bin/bash

chmod +x ./scripts/*

./scripts/setup_data.sh
./scripts/setup_weights.sh

./scripts/setup_preprocess_data.sh
