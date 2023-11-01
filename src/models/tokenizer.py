from sentencepiece import SentencePieceTrainer
import tempfile
import pandas as pd
import os
import shutil

def train(
    tsv_dataset_path: str, 
    max_pieces: int, 
    extra_pieces: list[str],
    save_path: str,
):
    """Train a sentencepiece tokenizer on the filtered paraphrase dataset.
    
    Args:
        tsv_dataset_path (str): path to the filtered paraphrase dataset
        max_pieces (int): Maximum number of tokens in the vocabulary
        extra_pieces (list[str]): Extra tokens to add to the vocabulary
        save_path (str): folder to save the model to
    """
    df = pd.read_csv(tsv_dataset_path, sep='\t')
    
    os.makedirs(save_path, exist_ok=True)
    
    with tempfile.NamedTemporaryFile('w+') as corpus:
        ref = df['reference'].to_list()
        trans = df['translation'].to_list()
        corpus.write('\n'.join(ref + trans))
        
        # train the tokenizer
        SentencePieceTrainer.train(
            input=corpus.name,
            model_prefix='tokenizer',
            vocab_size=max_pieces,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=extra_pieces
        )
        
        # save the tokenizer
        shutil.move('tokenizer.model', save_path)
        shutil.move('tokenizer.vocab', save_path)
        
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a sentencepiece tokenizer on the filtered paraphrase dataset.',
        usage='python tokenizer.py -f models/tokenizer -m 10000',
    )
    
    parser.add_argument(
        '--dataset', '-d', 
        type=str, 
        default='data/filtered.tsv', 
        help='path to the filtered paraphrase dataset'
    )
    
    parser.add_argument(
        '--file', '-f', 
        type=str, 
        default='data', 
        help='folder to save the model to'
    )
    
    parser.add_argument(
        '--max_pieces', '-m', 
        type=int, 
        default=1024, 
        help='Maximum number of tokens in the vocabulary'
    )
    
    # python ...
    # -e 0 <TOKEN!>
    # -e 1 <ANOTHER_TOKEN!>
    parser.add_argument(
        '--extra_pieces', '-e', 
        type=str, 
        nargs='+', 
        help='Extra tokens to add to the vocabulary'
    )
    
    args = parser.parse_args()

    train(
        args.dataset, 
        args.max_pieces, 
        args.extra_pieces,
        args.file,
    )
    
    print('\n\nOK.')
    