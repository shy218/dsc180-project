import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# def train(data_dir):
#     all_data = pd.read_pickle('../data/processed/feature_encoded_merged_data.pkl')
#
#     all_data.shape
