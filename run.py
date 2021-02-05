from os import listdir
import sys
from tqdm import tqdm
import pandas as pd

sys.path.insert(1, './src/')
from data_preprocessing import *
from data_downloads import *

def data_prep():
    # Check if raw files are ready
    if 'processed' not in listdir('data'):
        os.system('mkdir data/processed/')
    if 'raw' not in listdir('data/'):
        os.system('mkdir data/raw/')
    if '8K-gz' not in listdir('data/raw') or 'EPS' not in listdir('data/raw') \
        or 'price_history' not in listdir('data/raw'):
        to_dir = './data/raw/'
        data_download(to_dir)
    print(' => All raw data ready!')

    # Run part 1, 2
    raw_8k_fp = 'data/raw/8K-gz/'
    handler_clean_8k(raw_8k_fp)
    raw_eps_fp = 'data/raw/EPS/'
    handle_process_eps(raw_eps_fp)

    # Run part 3, 4
    handle_merge_eps8k_pricehist()


def main():
    if len(sys.argv) == 1:
        target = 'all'
    else:
        target = sys.argv[1]

    if target == 'data_prep':
        data_prep()


main()
