import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train(data_dir, out_dir):
    
    print()
    print('===================================================================')
    print(' => Training models...')
    print()
    
    data = pd.read_pickle('../data/processed/feature_encoded_merged_data.pkl')
    
    def select_phrases(phrases):
        return phrases[:2107]
    
    data['top_phrases'] = data['phrase_vec'].apply(select_phrases)
    
    train = data.loc[data['dataset'] == 'train'].copy()
    
    def create_model(all_data, train, **kwargs):

        num_train = train[['Surprise(%)', 'price_change_7', 
                  'price_change_30', 'price_change_90', 'price_change_365',
                  'prev_vix_values']].to_numpy()
        
        scaler = StandardScaler()
        scaler.fit(num_train)
        num_train = scaler.transform(num_train)

        mlb = MultiLabelBinarizer()
        all_events = pd.DataFrame(mlb.fit_transform(all_data['cleaned_event']),
                                  columns = mlb.classes_,
                                  index = all_data['cleaned_event'].index)   
        train_events = all_events.iloc[all_data.loc[all_data['dataset'] == 'train'].index].to_numpy()

        train_y = train[['target']].to_numpy().ravel()

        if kwargs['train_type'] == 'unigram':
            train_unigrams = np.array(train['unigram_vec'].values.tolist())
            train_X = np.concatenate((train_events, num_train, train_unigrams), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000, max_features = 1250)
            model = model.fit(train_X, train_y)

        if kwargs['train_type'] == 'phrase':
            train_phrases = np.array(train['top_phrases'].values.tolist())
            train_X = np.concatenate((train_events, num_train, train_phrases), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000, max_features = 1250)
            model = model.fit(train_X, train_y)
            
        if kwargs['train_type'] == 'base':
            train_X = np.concatenate((train_events, num_train), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000)
            model = model.fit(train_X, train_y)

        return model
    
    print('  => Training baseline model...')
    print()
    
    base_model = create_model(data, train, train_type = 'base')
    
    print()
    print('  => Training unigram model...')
    print()
    
    uni_model = create_model(data, train, train_type = 'unigram')
    
    print()
    print('  => Training phrase model...')
    print()
    
    phrase_model = create_model(data, train, train_type = 'phrase')
    
    print()
    print('  => Exporting models to pkl...')
    print()
    
    with open(out_dir, 'wb') as f:
        pickle.dump(base_model, f)
    with open(out_dir, 'wb') as f:
        pickle.dump(uni_model, f)
    with open(out_dir, 'wb') as f:
        pickle.dump(phrase_model, f)
    