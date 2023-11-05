import os
import pickle
import pandas as pd
import numpy as np
import torch
import argparse
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ToxicDataset(Dataset):
    def __init__(self, corpus, scores, max_len=512):
        self.corpus = corpus
        self.scores = scores
        self.max_len = max_len
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, index):
        sentence = self.corpus[index]
        sentence = np.array(sentence)
        sentence = np.pad(sentence, (0, self.max_len - len(sentence)), 'constant', constant_values=0)
        return sentence.astype(np.int32), self.scores[index].astype(np.float32)
    

class ToxicClassifier(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=False, 
            dropout=dropout, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.narrow = nn.Linear(output_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text))
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim)
        _, (hidden, _) = self.lstm(embedded)
        # (num_layers, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        hidden = hidden.sum(dim=0)
        # (batch_size, hidden_dim) -> (batch_size, output_dim)
        output = self.fc(hidden)
        # (batch_size, output_dim) -> (batch_size, 1)
        output = self.narrow(output)
        # (batch_size, 1) -> (batch_size)
        output = self.sigmoid(output).squeeze(1)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train_eval_lstm(
    corpus,
    scores, 
    validation_corpus,
    validation_scores,
    vocab_size, 
    embedding_dim, 
    hidden_dim, 
    output_dim, 
    n_layers, 
    dropout, 
    batch_size, 
    max_epochs
):
    data_loader = DataLoader(
        ToxicDataset(corpus, scores), batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    validation_loader = DataLoader(
        ToxicDataset(validation_corpus, validation_scores), batch_size=batch_size, num_workers=4
    )
    
    model = ToxicClassifier(
        vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout
    ).to(device)
    
    trainer = L.Trainer(
        accelerator='auto',
        max_epochs=max_epochs,
        val_check_interval=0.25,
        progress_bar_refresh_rate=1,
    )
    
    trainer.fit(model, data_loader, validation_loader)
    eval_ = trainer.validate(model, validation_loader)
    
    return model, eval_

def fine_tune_roberta(
    model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizer,
    corpus: list,
    scores: list,
    validation_corpus: list,
    validation_scores: list,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    output_dir: str,
):
    def tokenize(batch):
        inputs = tokenizer(
            batch['text'], 
            padding='longest', 
            truncation=True, 
            max_length=512
        )
        batch['input_ids'] = inputs['input_ids']
        batch['attention_mask'] = inputs['attention_mask']
        # Convert label to two labels
        batch['label'] = [1 if label > 0.5 else 0 for label in batch['label']]
        
        return batch
    
    train_dataset = pd.DataFrame({'text': corpus, 'label': scores})
    validation_dataset = pd.DataFrame({'text': validation_corpus, 'label': validation_scores})
    
    train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = Dataset.from_pandas(validation_dataset)
    
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=4)
    validation_dataset = validation_dataset.map(tokenize, batched=True, batch_size=batch_size, num_proc=4)
    
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    training_args = TrainingArguments(
        output_dir='output_dir',
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        report_to='none',
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # (*, 1)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            # (*, 2)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
            
    
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    return trainer, model, tokenizer, training_args
    
    
def get_frequencies(scores: list, corpus: list, vocab_size: int):
    total = 0
    freq = np.zeros(vocab_size)

    for tox, sentence in zip(scores, corpus):
        total += len(sentence)
        for word in sentence:
            freq[word] += tox * (-1 if tox < 0.5 else 1)
            
    freq = freq / total
    
    return freq


def train_naive(Xi, yi, freq: np.ndarray):
    alpha = nn.Parameter(torch.tensor(0.0).to(device))
    optimizer = torch.optim.Adam([alpha], lr=0.01)

    def train_step(X, y):
        optimizer.zero_grad()
        loss = torch.mean((y - alpha * torch.tensor(X)) ** 2)
        loss.backward()
        optimizer.step()
        return loss.item() 
    
    for j in trange(100):
        shuffle = np.random.permutation(len(Xi))
        Xi = [Xi[i] for i in shuffle]
        yi = [yi[i] for i in shuffle]
        for i in range(0, len(Xi), 10000):
            batch = Xi[i:i+1024]
            batch = [torch.tensor(x).mean() for x in batch]
            batch = torch.stack(batch).to(device)
            labels = torch.tensor(yi[i:i+1024]).to(device)
            loss = train_step(batch, labels)
            
            if i % (len(Xi) // 4) == 0:
                print(f"Epoch {j} Loss: {loss}")
                
    return alpha.item()


def eval_naive(alpha, X, y):
    preds = []
    for i in range(0, len(X), 128):
        batch = X[i:i+128]
        batch = [torch.tensor(x).mean() for x in batch]
        batch = torch.stack(batch)
        batch = batch.to(device)
        preds.append(alpha * batch)
    preds = torch.cat(preds).cpu().detach().numpy()
    preds = np.array([1 if pred > 0.5 else 0 for pred in preds])
    y = np.round(y)
    return roc_auc_score(y, preds), accuracy_score(y, preds), f1_score(y, preds), precision_score(y, preds), recall_score(y, preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='roberta')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--vocab_size', type=int, default=50265)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1E-5)
    parser.add_argument('--output_dir', type=str, default='models')
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join(args.data_dir, 'raw', 'filtered.tsv'), sep='\t', index_col=0)
    
    train = pickle.load(open(os.path.join(args.data_dir, 'interim', 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(args.data_dir, 'interim', 'val.pkl'), 'rb'))
    test = pickle.load(open(os.path.join(args.data_dir, 'interim', 'test.pkl'), 'rb'))
    
    train_df = df.iloc[train['indices']]
    val_df = df.iloc[val['indices']]
    test_df = df.iloc[test['indices']]
    
    corpus = train_df['reference'].tolist() + train_df['translation'].tolist()
    scores = np.concatenate([train['ref_tox'], train['trn_tox']])
    
    validation_corpus = val_df['reference'].tolist() + val_df['translation'].tolist()
    validation_scores = np.concatenate([val['ref_tox'], val['trn_tox']])
    

    
    if args.model == 'roberta' or args.model == 'all':
        tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
        trainer, model, tokenizer, training_args = fine_tune_roberta(
            model, tokenizer, corpus, scores, validation_corpus, validation_scores, 
            args.batch_size, args.max_epochs, args.learning_rate, args.output_dir
        )
        trainer.save_model(os.path.join(args.output_dir, 'classifier.best.model'))

    if args.model == 'lstm' or args.model == 'all':
        model, eval_ = train_eval_lstm(
            corpus, scores, validation_corpus, validation_scores, args.vocab_size, 
            args.embedding_dim, args.hidden_dim, args.output_dim, args.n_layers, 
            args.dropout, args.batch_size, args.max_epochs
        )
        
    if args.model == 'naive' or args.model == 'all':
        alpha = train_naive(corpus, scores, args.vocab_size)
        print(eval_naive(alpha, validation_corpus, validation_scores))
        