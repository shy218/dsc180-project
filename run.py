import sys
import json
import pandas as pd
from tqdm import tqdm
from os import listdir

sys.path.insert(1, './src/')
from data_preprocessing import *
from data_downloads import *
from feature_encoding import *

data_prep_config = json.load(open('config/data_prep.json', 'r'))
feature_encoding_config = json.load(open('config/feature_encoding.json', 'r'))
test_config = json.load(open('config/test.json', 'r'))

def data_prep(data_prep_config):
    data_dir = data_prep_config['data_dir']
    raw_dir = data_prep_config['raw_dir']

    # Download RAW data if needed (and if not in testing mode)
    if not data_prep_config['testing']:
        if 'raw' not in listdir(data_dir):
            os.system('mkdir ' + raw_dir)
        if '8K-gz' not in listdir(raw_dir):
            download_8k(raw_dir)
        if 'EPS' not in listdir(raw_dir):
            download_price_history(raw_dir)
        if 'price_history' not in listdir(raw_dir):
            download_eps(raw_dir)
        print(' => All raw data ready!')

    # Process 8K, EPS and Price History as needed
    processed_dir = data_prep_config['processed_dir']
    if 'processed' not in listdir(data_dir):
        os.system('mkdir ' + processed_dir)

    handler_clean_8k(data_prep_config['data_dir'])
    if not data_prep_config['testing']: # only process eps when it's not testing
        handler_process_eps(data_prep_config['data_dir'])
    print(' => Done 8k and eps cleaning!')
    # Run part 3, 4
    handle_merge_eps8k_pricehist(data_prep_config['data_dir'])

def feature_encoding():
    text_encode(**feature_encoding_config)

def main():
    if len(sys.argv) == 1:
        target = 'all'
    else:
        target = sys.argv[1]

    if target == 'data_prep':
        data_prep(data_prep_config)
    elif target == 'feature_encoding':
        feature_encoding()
    elif target == 'test':
        data_prep(test_config)

main()
