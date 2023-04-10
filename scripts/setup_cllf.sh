#!/bin/bash
# This script is used to download the model files from huggingface.co

git lfs install
git clone https://huggingface.co/yikuan8/Clinical-Longformer ./weights/Clinical-Longformer
