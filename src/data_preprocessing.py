import pandas as pd
import numpy as np
import datetime
from os import listdir
from bs4 import BeautifulSoup
from tqdm import tqdm

# Part 1 - Clean 8K reports
def clean_doc_text(doc_text):
    doc_text = re.sub('\n+', '\n', doc_text)
    cleaned_text = ''
    for sent in doc_text.split('\n'):
        cleaned_text += re.sub('\s+', ' ', sent)
        cleaned_text += '\n'
    doc_text = re.sub('\\t', ' ', doc_text)
    return doc_text

def handle_single_document(doc, all_failed_doc):
    time, event_type = None, None
    if doc.strip() == '\n' or doc == '\n' or doc == '':
        pass
    elif 'EVENTS:' not in doc or 'TIME:' not in doc:
        all_failed_doc.append(doc)
    else:
        time = re.findall('TIME:.+\\n', doc)[0]
        event_type = re.findall('EVENTS:.+\\n', doc)[0]
    return time, event_type

def handler_clean_8k(raw_8k_fp):
    print('===================================================================')
    print(' => Cleaning 8K...')
    all_data_dict = {}
    for fp in tqdm(listdir(raw_8k_fp)):
        if fp == '.DS_Store':
            continue
        print(fp)
        full_path = raw_8k_fp + fp
        file = open(full_path, 'r')
        tmp_txt = ''
        for line in file:
            tmp_txt += line

        all_failed_doc = []
        tmp_all_docs = []
        for doc in tmp_txt.split('</DOCUMENT>'):
            doc = doc.replace('</DOCUMENT>', '')
            time, event_type = handle_single_document(doc, all_failed_doc)
            if time:
                tmp_dict = {}
                tmp_dict['time'] = time
                tmp_dict['event_type'] = event_type
                tmp_dict['full_text'] = clean_doc_text(doc)
                tmp_all_docs.append(tmp_dict)
        all_data_dict[fp] = tmp_all_docs

    print(' => Saving the cleaned 8K report to local dir: data/processed/8k.json')
    with open('./data/processed/8k.json', 'w') as outfile:
        json.dump(all_data_dict, outfile)
    print(' => Example of one of the cleaned 8k report:', all_data_dict['AAPL'][0])
    print()
    print()
    return all_data_dict

# Part 2 - Process EPS
def get_EPS(file):
    result = []
    soup = BeautifulSoup(open(file), "html.parser")
    lst = soup.find_all('small')
    for tag in lst:
        a = tag.find('a')
        if a != None and a.text.isupper():
            temp = [a.text]
            p = a.parent
            for i in range(3):
                p = p.findNext('small')
                temp.append(p.text)
            result.append(temp)
    return result
def handler_process_eps(raw_eps_fp):
    print('===================================================================')
    print(' => Processing EPS...')
    result = []
    for file in listdir():
        print(file)
        temp = get_EPS(raw_eps_fp + file)
        date = int(file.split('.')[0])
        for t in temp:
            result.append([date] + t)
    df = pd.DataFrame(result, columns = ['Report Date', 'Code', 'Surprise(%)', 'Reported EPS', 'Consensus EPS'])
    print(' => Saving the processed EPS infomation to local dir: data/processed/EPS.csv')
    df.to_csv('./data/processed/EPS.csv', index = None)

# Part 3 - Merge EPS and 8K text
def clean_time(time_str):
    time_str = re.sub('[^0-9]', '', time_str)
    date = time_str[:8]
    time = time_str[8:]
    return date, time

def merge_EPS_8K():
    with open('../data/processed/8k.json') as json_file:
        all_data_dict = json.load(json_file)
    # all_data_dict = handler_clean_8k() # output of step 1

    all_8k_lst = []
    doc_counter = 0
    for symbol in tqdm(all_data_dict.keys()):
        tmp_docs = all_data_dict[symbol]
        for doc in tmp_docs:
            doc_counter += 1
            date, time = clean_time(doc['time'])
            all_8k_lst.append({
                'date': date,
                'time': time,
                'event_type': doc['event_type'],
                'full_text': doc['full_text'],
                'Code': symbol
            })
    all_8k_df = pd.DataFrame(all_8k_lst)
    all_8k_df['time_code'] = all_8k_df.date + all_8k_df.Code

    eps_df = pd.read_csv('./data/processed/EPS.csv') # output of step 2
    eps_df['time_code'] = eps_df['Report Date'].apply(lambda x: str(x)) + eps_df.Code

    merged_df = all_8k_df.merge(eps_df, how = 'inner', on = 'time_code').dropna()
    merged_df = merged_df.drop(columns = ['Code_x', 'Report Date', 'time_code'])\
        .rename(columns = {'Code_y': 'symbol'})
    merged_df['hr'] = merged_df.time.apply(lambda time: float(time[:4]) / 100)
    merged_df['pre_market'] = merged_df['hr'] < 9.5
    merged_df.date = merged_df.date\
        .apply(lambda x: pd.to_datetime(x[:4] + '-' + x[4:6] + '-' + x[6:]))
    return merged_df

# Part 4
def calc_price_changes(price_df, date_delta, intended_dates):
    price_delta = []
    for date_idx in intended_dates:
        if date_idx == -1:
            price_delta.append(np.nan)
            continue
        # Check if previous trading date is logged
        prev_date = date_idx - date_delta
        while prev_date not in price_df.date_idx.values and prev_date >= 0:
            prev_date -= 1
        if prev_date < 0:
            price_delta.append(np.nan)
            continue

        # Find prev info
        prev_close = price_df.query('date_idx == "' + str(prev_date) + '"')['Adj Close'].values[0]
        curr_close = price_df.query('date_idx == "' + str(date_idx) + '"')['Adj Close'].values[0]
        percent_change = round((curr_close - prev_close) / prev_close * 100, 2)
        price_delta.append(percent_change)
    return price_delta

def handle_merge_eps8k_pricehist():
    merged_df = merge_EPS_8K() # Call part 3 code
    min_date = merged_df.date.min()
    merged_df['date_idx'] = merged_df.date.apply(lambda x: (x - min_date).days)
    max_date_idx = merged_df['date_idx'].max()

    symbol_missing_price = []
    sub_dfs = []
    for symbol in tqdm(merged_df.symbol.unique()):
        try:
            price_hist_df = pd.read_csv(price_history_dir + symbol + '.csv')
        except:
            symbol_missing_price.append(symbol)
            continue

        # pre-process price_history
        price_hist_df.Date = price_hist_df.Date.apply(lambda x: (pd.to_datetime(x)))
        price_hist_df['date_idx'] = price_hist_df.Date.apply(lambda x: (x - min_date).days)

        # Get intended dates
        tmp_merged_df = merged_df.query('symbol == "' + symbol + '"').reset_index(drop = True)
        # intended_dates = tmp_merged_df.apply(lambda s: s.date_idx if s.pre_market else s.date_idx + 1)
        intended_dates = []
        for index, row in tmp_merged_df.iterrows():
            d_idx = row.date_idx
            if not row.pre_market:
                d_idx += 1
            while d_idx not in price_hist_df.date_idx.values and d_idx <= max_date_idx:
                d_idx += 1
            if d_idx > max_date_idx:
                d_idx = -1
            intended_dates.append(d_idx)

        # find out price changes
        for date_delta in [7, 30, 90, 365]:
            tmp_price_changes = calc_price_changes(price_hist_df, date_delta, intended_dates)
            tmp_merged_df['price_change_' + str(date_delta)] = tmp_price_changes
        sub_dfs.append(tmp_merged_df)

    updated_merged_df = pd.concat(sub_dfs)
    updated_merged_df.dropna().to_csv('./data/processed/merged_all_data.csv', index = False)
    return 0

# def calc_price_changes(tmp_price_df, date_delta):
#     price_delta = []
#     for date in (tmp_price_df.time_idx.values):
#         # Check if previous trading date is logged
#         prev_date = date - date_delta
#         while prev_date not in tmp_price_df.time_idx.values and prev_date >= 0:
#             prev_date -= 1
#         if prev_date < 0:
#             price_delta.append(np.nan)
#             continue
#
#         # Find prev info
#         prev_close = tmp_price_df.query('time_idx == "' + str(prev_date) + '"')['Adj Close'].values[0]
#         curr_close = tmp_price_df.query('time_idx == "' + str(date) + '"')['Adj Close'].values[0]
#         percent_change = round((curr_close - prev_close) / prev_close * 100, 2)
#         price_delta.append(percent_change)
#     return price_delta
#
# def process_single_company(tmp_price_df):
#     # Clean date data
#     tmp_price_df.Date = tmp_price_df.Date.apply(lambda x: pd.to_datetime(x))
#     min_date = min(tmp_price_df.Date.values)
#     tmp_price_df['time_idx'] = tmp_price_df.Date.apply(lambda x: (x - min_date).days)
#
#     # Calc corresponding fields
#     for date_delta in [7, 30, 90, 365]:
#         tmp_price_df['price_change_' + str(date_delta)] = calc_price_changes(tmp_price_df, date_delta)
#     return tmp_price_df
