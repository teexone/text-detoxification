import argparse
import pickle 
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import os
import pandas as pd
import numpy as np

def prepare_data(
    data_folder: str,
    model_folder: str,
    vocab_size: int = 10000,
):
    """Prepare data for training and testing.
    
    Args:
        data_folder: Path to folder with data.
        model_folder: Path to folder with models.
        vocab_size: Size of vocabulary.
    """
    
    # Read data
    data = pd.read_csv(
        os.path.join(data_folder, 'raw', 'filtered.tsv'),
        sep='\t',
    )
    
    # Filter out empty translations
    corpus = data['reference'].to_list() + data['translation'].to_list()
    
    # Write corpus to file
    with open(os.path.join(data_folder, 'interim', 'corpus.txt'), 'w') as f:
        f.write('\n'.join(corpus))
        
    # Train SentencePiece model
    SentencePieceTrainer.Train(
        '--input={0} --model_prefix={1} --vocab_size={2}'.format(
            os.path.join(data_folder, 'interim', 'corpus.txt'),
            os.path.join(model_folder, 'spm'),
            vocab_size
        )
    )
    
    # Load SentencePiece model
    sp = SentencePieceProcessor()
    sp.Load(os.path.join(model_folder, 'spm.model'))
    
    # Encode data
    def encode(x):
        "<s> X </s>"
        return [sp.bos_id()] + sp.EncodeAsIds(x) + [sp.eos_id()]

    # Encode data
    data['reference'] = data['reference'].apply(encode)
    data['translation'] = data['translation'].apply(encode)
    
    # Let's tranform the dataset in
    # the way reference toxicity is always
    # greater than translation toxicity
    flipped_data = data.copy()

    flips = 0
    for i, row in data.iterrows():
        if row['ref_tox'] < row['trn_tox']:
            flipped_data.at[i, 'reference'], flipped_data.at[i, 'translation'] = data.at[i, 'translation'], data.at[i, 'reference']
            flipped_data.at[i, 'ref_tox'], flipped_data.at[i, 'trn_tox'] = data.at[i, 'trn_tox'], data.at[i, 'ref_tox']
            flips += 1
        
    print("Flips: {}".format(flips))
    
    assert all(flipped_data['ref_tox'] >= flipped_data['trn_tox'])
    
    # Compress data
    reference_array = [np.array(f, dtype=np.uint16) for f in flipped_data['reference'].values]
    translation_array = [np.array(f, dtype=np.uint16) for f in flipped_data['translation'].values]
    ref_tox = flipped_data['ref_tox'].to_numpy().astype(np.float32)
    trn_tox = flipped_data['trn_tox'].to_numpy().astype(np.float32)

    assert len(reference_array) == len(translation_array) == ref_tox.shape[0] == trn_tox.shape[0]

    np.random.seed(1409)
    
    # Split data
    train_indices = np.random.choice(len(reference_array), int(len(reference_array) * 0.8), replace=False)
    val_indices = np.setdiff1d(np.arange(len(reference_array)), train_indices)
    test_indices = np.random.choice(val_indices, int(len(val_indices) * 0.5), replace=False)
    val_indices = np.setdiff1d(val_indices, test_indices)

    assert len(train_indices) + len(val_indices) + len(test_indices) == len(reference_array)
    

    with open(os.path.join(data_folder, 'interim', 'train.pkl'), 'wb') as f:
        refs = [reference_array[i] for i in train_indices]
        trns = [translation_array[i] for i in train_indices]
        pickle.dump(
            {
                'reference': refs,
                'translation': trns,
                'ref_tox': ref_tox[train_indices],
                'trn_tox': trn_tox[train_indices],
                'indices': train_indices
            },
            f
        )
        
    with open(os.path.join(data_folder, 'interim', 'val.pkl'), 'wb') as f:
        refs = [reference_array[i] for i in val_indices]
        trns = [translation_array[i] for i in val_indices]
        pickle.dump(
            {
                'reference': refs,
                'translation': trns,
                'ref_tox': ref_tox[val_indices],
                'trn_tox': trn_tox[val_indices],
                'indices': val_indices
            },
            f
        )
        
        
    with open(os.path.join(data_folder, 'interim', 'test.pkl'), 'wb') as f:
        refs = [reference_array[i] for i in test_indices]
        trns = [translation_array[i] for i in test_indices]
        pickle.dump(
            {
                'reference': refs,
                'translation': trns,
                'ref_tox': ref_tox[test_indices],
                'trn_tox': trn_tox[test_indices],
                'indices': test_indices
            },
            f
        )
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='data')
    parser.add_argument('--model-folder', type=str, default='models')
    parser.add_argument('--vocab-size', type=int, default=10000)
    args = parser.parse_args()
    prepare_data(
        data_folder=args.data_folder,
        model_folder=args.model_folder,
        vocab_size=args.vocab_size
    )
        