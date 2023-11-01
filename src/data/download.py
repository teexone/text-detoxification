import os
import requests
import zipfile
import tempfile

def download(save_folder: str):
    """Download the filtered paraphrase dataset from the github release.

    Args:
        save_folder (str): folder to save the dataset in

    Returns:
        str: path to the filtered paraphrase dataset
    """
    zip_ = requests.get(
        'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
    )
    
    # zip file contains
    # a single file
    # filtered.tsv
    with tempfile.TemporaryFile() as f:
        f.write(zip_.content)
        with zipfile.ZipFile(f) as z:
            z.extractall(save_folder)
    
    assert os.path.exists(os.path.join(save_folder, 'filtered.tsv'))
    
    return os.path.join(save_folder, 'filtered.tsv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Download the filtered paraphrase dataset from the github release.',
        usage='python download.py -f data',
    )

    parser.add_argument(
        '--file', '-f', 
        type=str, 
        default='data', 
        help='folder to save the dataset in'
    )
    
    args = parser.parse_args()
    
    download(args.f)
    
    print('\n\nOK.')