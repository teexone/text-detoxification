Aleksandr Lobanov, B20-AAI, a.lobanov@innopolis.university

# Detoxification

## Preface

The following is a report on my exploration of possible solution for text detoxification. The report contains raw, unmaterialized ideas and is not meant to be a final product.

## Introduction

Detoxification is the process of replacing toxic elements from a text. Toxic elements are considered to be words or phrases that are considered offensive, rude, or otherwise undesirable. The goal of detoxification is to replace toxic elements with neutral or positive ones, while preserving the original meaning of the text.

### Problem

The problem of detoxification is a problem of **text generation**. The goal is to generate a text that is _similar_ to the original text, but does not contain toxic elements. Apart from the quality of the generated text, the main challenge is to define the toxicity of a phrase and whether generated text is actually a less toxic version of the original text. 


### Ideas

As described in [1], the problem of detoxification can be approached in two ways: **rephrasing** and **retrieval**. Rephrasing is the process of generating a new text from scratch, while retrieval is the process of finding a text that is similar to the original text, but does not contain toxic elements. Also, retrieval can be done on the level of words or phrases. 

#### Rephrasing

Rephrasing can be done using a **language model**. A language model is a model that is trained to predict the next word in a sequence of words. The model is trained on a large corpus of text, and is able to generate text that is similar to the text it was trained on. The model can be trained on a corpus of non-toxic text, and then used to generate a non-toxic version of the original text. This approach is also called 'parallel' as it requires a parallel corpus of toxic and non-toxic text. While [2] offers a solution to this problem, it exploits unsupervised learning, while I would like to explore supervised learning as I am provided

#### Retrieval

Retrieval can be done using a **similarity model**. A similarity model is a model that is trained to predict whether two texts are similar. The model is trained on a large corpus of text, and is able to predict whether two texts are similar. The model can be trained on a corpus of toxic and non-toxic text, and then used to find a non-toxic version of the original text. This approach does not require a parallel corpus of toxic and non-toxic text, but it does require much larger amount of samples to be able to find a similar text. I don't think this approach is feasible for this project, as we wish to preserve the original meaning of the text, and it is unlikely that a similar text will have the same meaning.

#### Retrieval on the level of words

However, retrieve can be performed on the level of words. But, this might descrease semantic and grammatic correctness of the text, as [2] states.

### Experiments

I would like to explore the following approach: train a classifier to distinguish toxic and non-toxic text and them train sequence-to-sequence to perform text translation task. 

For the classifier I tried to use a very naive frequency-based approach. Then I tried LSTM-based classifier, which showed slightly better results. At the end I fine-tuned RoBERTa [3] model, which showed the best results.

For the sequence-to-sequence model I decide not to experiment with my own architectures as they will probably underperform any transformer. So my choice is to fine-tune T5 [4] model on the ParaMNT dataset.

## Methods

### Classifier

#### Frequency-based classifier

The frequency-based classifier assigns a toxicity score to a text based on the frequency of toxic words in the text. The score is calculated as a sum of scores $s_i$ of each word $w_i$ in the text. The score $s_i$ is calculated as follows:

$$m_i = \frac{f_i}{\sum_{j=1} f_j}$$

$$s_i = -1 + 2m_i$$


where $f_i$ is the frequency of word $w_i$ in the corpus of toxic words.

The classifier is a single parameter alpha $\alpha$ that is used to calculate the toxicity score of the text. The score is calculated as follows:
$\hat{y} = \alpha \sum_{i=1}^n s_i$

#### LSTM-based classifier

The LSTM-based classifier is a simple LSTM-based model that takes a sequence of words as input and outputs a toxicity score.

#### RoBERTa-based classifier

The RoBERTa-based classifier is a RoBERTa-based model that takes a sequence of words as input and outputs a toxicity score. The model is fine-tuned on the dataset of toxic and non-toxic texts (ParaMNT).

### Sequence-to-sequence

The sequence-to-sequence model is a T5-based model that takes a sequence of words as input and outputs a sequence of words. The model is fine-tuned on the ParaMNT dataset.

### Metrics

The classifier is evaluated using the following metrics: accuracy, F1-score, precision, recall, ROC AUC. The sequence-to-sequence model is evaluated using compound metrics described in [2]: style accuracy, similarity accuracy, BLEU score and J metric, described in [3]

## Results

### Classifier

| Model | Accuracy | F1-score | Precision | Recall | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Frequency-based | 0.48 | 0.51 | 0.54 | 0.52 | 0.47 |
| LSTM-based | 0.56 | 0.56 | 0.53 | 0.51 | 0.49 |
| RoBERTa-based | 0.98 | 0.98 | 0.99 | 0.98 | 0.89 |

### Sequence-to-sequence

| Model | ACC | SIM | FL | J | BLEU |

| ----- | --- | --- | -- | - | ---- |

T5|0.6359|0.9795|0.9144|0.5674|0.9214|

### Examples

| Original | Detoxified |
| --- | --- |
When he's out, you're packing up and you're keeping the fuck away from my sister! | When he leaves, you pack up and get away from my sister!
| Shut up, Sam. | Hush, Sam.
|  Amazing tits. | Beautiful breasts...



## Conclusion

The results of the experiments show that the proposed approach is not feasible. The classifier is not able to distinguish toxic and non-toxic texts, and the sequence-to-sequence model is not able to generate a detoxified version of the original text.

