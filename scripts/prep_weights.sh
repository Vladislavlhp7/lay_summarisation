#!/bin/bash

# Load environment variables
source .env

# Set the download URL
DOWNLOAD_URL="https://physionet.org/files/clinical-t5/1.0.0/"

# Set the output directory
OUTPUT_DIR="weights"
mkdir -p $OUTPUT_DIR

# Download files using wget command
wget -r -N -c -np --user $USERNAME --password=$PASSWORD $DOWNLOAD_URL -P $OUTPUT_DIR
