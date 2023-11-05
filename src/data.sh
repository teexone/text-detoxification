#!/bin/bash

# Check if we are in
# the correct directory
if [ ! -f "src/data.sh" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi

# Check if the data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory"
    mkdir data
fi

# Check if the data subdirectories exist
if [ ! -d "data/external" ]; then
    echo "Creating data/external directory"
    mkdir -p data/external
fi

if [ ! -d "data/interim" ]; then
    echo "Creating data/interim directory"
    mkdir -p data/interim
fi

if [ ! -d "data/raw" ]; then
    echo "Creating data/raw directory"
    mkdir -p data/raw
fi

# Check Python
if ! command -v python &> /dev/null
then
    echo "Python could not be found"
    exit 1
fi

# Check Python version
if ! python -c 'import sys; exit(not (sys.version_info.major == 3 and 6 <= sys.version_info.minor <= 9))'
then
    echo "Required 3.6<= Python <=3.9"
    exit 1
fi

pip install -r requirements.txt
echo "[OK. ] Python"

# Download data and exit if it fails
python src/data/download.py
if [ $? -ne 0 ]; then
    echo "[FAIL] Downloading data"
    exit 1
fi

echo "[OK. ] Downloading data"

python src/data/make_dataset.py --data-folder data --model-folder models --vocab-size 10000
if [ $? -ne 0 ]; then
    echo "[FAIL] Making dataset"
    exit 1
fi

echo "[OK. ] Making dataset"




