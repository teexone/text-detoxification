# Detoxification

## Description

The repository contains experiments related to detoxification task. The task is to predict the toxicity of a comment and rewrite it in a non-toxic way. The repository contains the following files:

- `data/` - folder with data, run src/data.sh to download the data
- `src/` - folder with source code
- `models/` - folder with trained models
- `notebooks/` - folder with results of the experiments

## Data

The repository uses only ParaMNT dataset

## Setup

The repository uses Python 3.9. To install the dependencies run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train the model run the following command:

```bash
python src/models/classifier.py --help
```

To train seq2seq model run the following command:

```bash
python src/models/seq2seq.py --help
```

## Evaluation

To evaluate the model run the following command:

```bash
python src/models/seq2seq.py --do-eval
```

To run metric evaluation from "Text Detoxification using Large Pre-trained Language Models" paper run the following command:

```bash
bash src/models/eval.sh data/interim/references.txt data/interim/predictions.txt
```