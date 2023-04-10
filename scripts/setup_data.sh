#!/bin/bash

# Set the download URL and output file name
DOWNLOAD_URL="https://drive.google.com/uc?id=1FFfa4fHlhEAyJZIM2Ue-AR6Noe9gOJOF&export=download&confirm=t&uuid=01b721da-12a0-46cd-a5b4-9ce1620a4d1e"
OUTPUT_FILE="./data/task1_development.zip"

# Set the output directory for extracted files
OUTPUT_DIR="./data/orig"
mkdir -p $OUTPUT_DIR

# cleanup
rm -rf $OUTPUT_DIR

# Download the file using curl command
curl -L -o $OUTPUT_FILE $DOWNLOAD_URL

# Extract the contents of the zip file
unzip $OUTPUT_FILE -d $OUTPUT_DIR

# Delete the zip file
rm $OUTPUT_FILE
