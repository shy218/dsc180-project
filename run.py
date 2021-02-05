from os import listdir
import sys
from tqdm import tqdm
import pandas as pd

sys.path.insert(1, './src/')
from features import *

def handle_price_change():
    all_cleaned = []
    file_lst = listdir('data/price_history/')
    counter = 0
    for comp in tqdm(file_lst):
        if 'csv' in comp:
            cleaned_price = process_single_company(pd.read_csv('data/price_history/' + comp))
            cleaned_price['company'] = comp.replace('.csv', '')
            all_cleaned.append(cleaned_price)
            counter += 1
        # if counter > 3:
        #     break
    all_cleaned_df = pd.concat(all_cleaned)
    all_cleaned_df.to_csv('./data/processed/price_history_all_cleaned.csv', index = False)
    return 0

def main():
    if len(sys.argv) == 1:
        target = 'all'
    else:
        target = sys.argv[1]

    if target == 'features':
        handle_price_change()




main()
