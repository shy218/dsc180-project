import sys
import json
import pandas as pd
from tqdm import tqdm
from os import listdir

sys.path.insert(1, './src/')
from data_preprocessing import *
from data_downloads import *
from feature_encoding import *

data_prep_config = json.loads(open('config/data_prep.json'))
feature_encoding_config = json.loads(open('config/feature_encoding.json'))

def data_prep():
    # Check if raw files are ready
    if 'processed' not in listdir('data'):
        os.system('mkdir data/processed/')
    if 'raw' not in listdir('data/'):
        os.system('mkdir data/raw/')
    if '8K-gz' not in listdir('data/raw') or 'EPS' not in listdir('data/raw') \
        or 'price_history' not in listdir('data/raw'):
        # to_dir = './data/raw/'
        data_download(data_prep_config['to_dir'])
    print(' => All raw data ready!')

    # Run part 1, 2
    # raw_8k_fp = 'data/raw/8K-gz/'
    handler_clean_8k(data_prep_config['raw_8k_fp'])
    # raw_eps_fp = 'data/raw/EPS/'
    handler_process_eps(data_prep_config['raw_eps_fp'])
    print(' => Done 8k and eps cleaning!')
    # Run part 3, 4
    handle_merge_eps8k_pricehist()

def feature_encoding():
    text_encode(**feature_encoding_config)

def main():
    if len(sys.argv) == 1:
        target = 'all'
    else:
        target = sys.argv[1]

    if target == 'data_prep':
        data_prep()
    elif target == 'feature_encoding':
        feature_encoding()

main()
