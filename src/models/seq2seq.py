import os
import pandas as pd
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
import numpy as np
import torch 
from datasets import Dataset
from sklearn.model_selection import train_test_split


DATA_FOLDER = os.path.join(os.getcwd(), 'data')
DATASET_FILE = os.path.join(DATA_FOLDER, 'raw', 'filtered.tsv')
MODEL_FOLDER = os.path.join(os.getcwd(), 'models')
MODEL_PREFIX = os.path.join(MODEL_FOLDER, 'tokenizer')
VOCAB_SIZE = 10000 # spiece

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(
    batch_size = 35,
    num_epochs = 1,
    learning_rate = 5e-5,
    warmup_steps = 500,
    weight_decay = 0.01,
    source_train = None,
    target_train = None,
    source_val = None,
    target_val = None,
):
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('t5-small').to(device)
    
    def tokenize_function(examples):
        """Tokenize the examples
        
        :param examples: the examples to tokenize
        
        :return: the tokenized examples
        """
        inputs = tokenizer.batch_encode_plus(
            examples['translation'], 
            padding='max_length',
            max_length=512,
            add_special_tokens=True,
            truncation=True,
        )
        
        labels = tokenizer.batch_encode_plus(
            examples['reference'], 
            padding='max_length',
            max_length=512,
            add_special_tokens=True,
            truncation=True,
        ).input_ids
        
        labels_with_ignore_index = []
        for labels_example in labels:
            # Replace 0 with -100 (T5 default ignore index)
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        
        inputs['labels'] = labels_with_ignore_index
        
        return inputs
    
    train_dataset = Dataset.from_dict({'translation': source_train, 'reference': target_train})
    val_dataset = Dataset.from_dict({'translation': source_val, 'reference': target_val})

    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=512, num_proc=6)
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=512, num_proc=6)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_FOLDER,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=1000,
        save_steps=1000,
        eval_steps=1000,
        overwrite_output_dir=True,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    return model, tokenizer 



def eval(
    model_folder: str = MODEL_FOLDER,
    model_name: str = 't5-detox-best',
    save_references_path: str = os.path.join(DATA_FOLDER, 'interim', 'references.txt'),
    save_predictions_path: str = os.path.join(DATA_FOLDER, 'interim', 'predictions.txt'),
    save_input_texts_path: str = os.path.join(DATA_FOLDER, 'interim', 'inputs.txt'),
):
    pd_data = pd.read_csv(os.path.join(DATA_FOLDER, 'raw', 'filtered.tsv'), sep='\t')

    source = pd_data['translation'].tolist()
    target = pd_data['reference'].tolist()
    
    torch.manual_seed(705)
    np.random.seed(705)

    source_val_train, source_test, target_val_train, target_test = train_test_split(source, target, test_size=0.2)
    source_train, source_val, target_train, target_val = train_test_split(source_val_train, target_val_train, test_size=0.2)
        
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(model_folder, model_name)).to(device)
    tokenizer = T5Tokenizer.from_pretrained(os.path.join(model_folder, model_name))
 
    test_dataset = Dataset.from_dict({'translation': source_test, 'reference': target_test})
    def tokenize_function(examples):
        """
        Tokenize the examples
        
        :param examples: the examples to tokenize
        
        :return: the tokenized examples
        """
        inputs = tokenizer.batch_encode_plus(
            examples['translation'], 
            padding='max_length',
            max_length=512,
            add_special_tokens=True,
            truncation=True,
        )
        
        labels = tokenizer.batch_encode_plus(
            examples['reference'], 
            padding='max_length',
            max_length=512,
            add_special_tokens=True,
            truncation=True,
        ).input_ids
        
        labels_with_ignore_index = []
        for labels_example in labels:
            # Replace 0 with -100 (T5 default ignore index)
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        
        inputs['labels'] = labels_with_ignore_index
        return inputs

    test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=512, num_proc=6)
    
    def generate_translation(batch):
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            early_stopping=True,
        )
        
        return outputs
    
    input_texts = []
    predictions = []
    references = []

    def gen(batch):
        input_texts.extend(batch['reference'])
        outputs = generate_translation(batch)
        
        predictions.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])
        references.extend(batch['translation'])
        
    test_dataset.select(range(10000)).map(gen, batched=True, batch_size=32)
    
    with open(save_references_path, 'w+') as f:
        f.write('\n'.join(references))
    
    with open(save_predictions_path, 'w+') as f:
        f.write('\n'.join(predictions))
        
    with open(save_input_texts_path, 'w+') as f:
        f.write('\n'.join(input_texts))
        
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--do-training', action='store_true')
    parser.add_argument('--batch-size', type=int, default=35)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--model-folder', type=str, default=MODEL_FOLDER)
    parser.add_argument('--model-name', type=str, default='t5-detox-best')
    parser.add_argument('--save-references-path', type=str, default=os.path.join(DATA_FOLDER, 'interim', 'references.txt'))
    parser.add_argument('--save-predictions-path', type=str, default=os.path.join(DATA_FOLDER, 'interim', 'predictions.txt'))
    parser.add_argument('--data-folder', type=str, default=DATA_FOLDER)
    parser.add_argument('--dataset-file', type=str, default=DATASET_FILE)
    
    args = parser.parse_args()
    
    MODEL_FOLDER = args.model_folder
    DATASET_FILE = args.dataset_file
    DATA_FOLDER = args.data_folder
    SAVE_REFERENCES_PATH = args.save_references_path
    SAVE_PREDICTIONS_PATH = args.save_predictions_path
    
    
    
    args = parser.parse_args()
    
    if args.do_eval:
        eval()
        
        
    