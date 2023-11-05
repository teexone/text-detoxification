#!/bin/bash

if [ ! -d "snlp" ]; then
    git clone https://github.com/s-nlp/detox snlp
fi

cd snlp
pip install gdown fairseq==0.10.0 "numpy<1.20.0"
cd emnlp2021
bash prepare.sh

python metric/metric.py -i $1 -p $2