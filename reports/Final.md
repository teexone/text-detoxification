Aleksandr Lobanov, B20-AAI, a.lobanov@innopolis.university

# Final

In the final results, the following models were used:

1. For the task of toxicity classification the RoBERTa [<a href="#user-content-1">1</a>, <a href="#user-content-2">2</a>] classifier was used. It was pre-trained on the compilation of Jigsaw datasets and fine-tuned on the train part of ParaMNT under regression task
2. For the task of rewriting toxic comments the T5 model was used. It was fine-tuned on the train part of ParaMNT under sequence-to-sequence task

## Toxicity classification

The RoBERTa [1, 2]  model was fine-tuned on the train part of ParaMNT under regression task. The model was trained for 3 epochs with the following parameters:

- Batch size: $32$
- Learning rate: $10^{-5}$
- Optimizer: AdamW
- Loss function: MSE


The model was evaluated on the validation part of ParaMNT. The results are presented in the table below:
<div align='center' style='min-width: 75%; font-size: 1.1em'>

| Metric | Value |
| --- | --- |
| Accuracy | 0.98 |
| Precision | 0.99 |
| Recall | 0.98 |
| F1 | 0.98 |
| ROC AUC | 0.89 |

</div>

## Rewriting toxic comments

The T5 model was fine-tuned on the train part of ParaMNT under sequence-to-sequence task. The model was trained for 2 epochs with the following parameters:

- Batch size: $35$
- Learning rate: $5\cdot10^{-5}$
- Optimizer: AdamW
- Loss function: CrossEntropyLoss

To evaluate the model, I used compound metrics from [<a href="#user-content-2">2</a>]: the BLEU score, which measures the similarity between the generated and the reference sentences, the SIM score, which measures the semantic similarity between the generated and the reference sentences using sentence embeddings, the ACC score, which measures the accuracy of style transfer between the generated sentences, the FL score, which measures the fluency of the generated sentences, and the J score [<a href="#user-content-3">3</a>] which is a combination of last three metrics.

The model was evaluated on the validation part of ParaMNT. The results are presented in the table below:
<div align='center' style='min-width: 75%; font-size: 1.1em'>

| Metric | Value |
| --- | --- |
| BLEU | 92.4 |
| SIM | 0.97 |
| ACC | 0.63 |
| FL | 0.91 |
| J | 0.5674 |

</div>

## References

<span id="1">[1]</span>. Liu, Y. et al. (2019) Roberta: A robustly optimized Bert pretraining approach, aclanthology.org. Available at: https://aclanthology.org/2021.ccl-1.108/ (Accessed: 01 November 2023). 

<span id="2">[2]</span>. Dale, D. et al. (no date) Text detoxification using large pre-trained neural models, ACL Anthology. Available at: https://aclanthology.org/2021.emnlp-main.629/ (Accessed: 03 November 2023). 

<span id="3">[3]</span>. Krishna, K., Wieting, J. and Iyyer, M. (no date) Reformulating unsupervised style transfer as paraphrase generation, ACL Anthology. Available at: https://aclanthology.org/2020.emnlp-main.55/ (Accessed: 05 November 2023). 
