import pandas as pd
import numpy as np
import tqdm
import datetime
from os import listdir

def calc_price_changes(tmp_price_df, date_delta):
    price_delta = []
    for date in (tmp_price_df.time_idx.values):
        # Check if previous trading date is logged
        prev_date = date - date_delta
        while prev_date not in tmp_price_df.time_idx.values and prev_date >= 0:
            prev_date -= 1
        if prev_date < 0:
            price_delta.append(np.nan)
            continue

        # Find prev info
        prev_close = tmp_price_df.query('time_idx == "' + str(prev_date) + '"')['Adj Close'].values[0]
        curr_close = tmp_price_df.query('time_idx == "' + str(date) + '"')['Adj Close'].values[0]
        percent_change = round((curr_close - prev_close) / prev_close * 100, 2)
        price_delta.append(percent_change)
    return price_delta

def process_single_company(tmp_price_df):
    # Clean date data
    tmp_price_df.Date = tmp_price_df.Date.apply(lambda x: pd.to_datetime(x))
    min_date = min(tmp_price_df.Date.values)
    tmp_price_df['time_idx'] = tmp_price_df.Date.apply(lambda x: (x - min_date).days)

    # Calc corresponding fields
    for date_delta in [7, 30, 90, 365]:
        tmp_price_df['price_change_' + str(date_delta)] = calc_price_changes(tmp_price_df, date_delta)
    return tmp_price_df
