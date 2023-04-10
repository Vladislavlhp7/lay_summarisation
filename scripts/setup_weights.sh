#!/bin/bash

mkdir -p weights

# Download the clinical longformer model weights
git lfs install
git clone https://huggingface.co/yikuan8/Clinical-Longformer ./weights/Clinical-Longformer
