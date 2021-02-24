import re
import heapq
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def text_encode(data_file, phrase_file, n_unigrams, threshhold, train_split, test_split, **kwargs):
    print()
    print('===================================================================')
    print(' => Text encoding...')
    print()
    
    merged_data = pd.read_csv(data_file)

    #Cleaned Event Feature

    def event_clean(text):
        result = re.sub('\n', '', text)
        result = re.sub('\t+', '\t', result)
        result = re.split('\t', result)

        if len(result) > 0:
            result = [s.lower() for s in result[1:]] # exclude the first item
            cleaned_result = []
            for s in result:
                if ';' in s:
                    for sub in s.split(';'):
                        cleaned_result.append(sub.strip())
                else:
                    cleaned_result.append(s.strip())
            return cleaned_result
        else:
            return ['Missing']

    cleaned_event = merged_data['event_type'].apply(event_clean)
    merged_data.insert(3, 'cleaned_event', cleaned_event)
    
    def event_clean_2(text):
        result = []
        for event in text:
            cleaned = event.replace('2.02', '').strip()
            if cleaned != '' and cleaned not in result:
                result.append(cleaned)
        return result

    merged_data['cleaned_event'] = merged_data['cleaned_event'].apply(event_clean_2)
    
    #Creating Target Variable
    
    def up_down_stay(price):
        if abs(price) < 1:
            return 'STAY'
        if price < 0:
            return 'DOWN'
        else:
            return 'UP'
    
    merged_data['target'] = merged_data['targe_price_change'].apply(up_down_stay)
    
    #Train, Val, Test Split for Encoding

    X_train, X_test = train_test_split(merged_data, test_size = 1 - train_split, random_state = 42)
    X_val, X_test = train_test_split(X_test, test_size = test_split / (1 - train_split), random_state = 42)

    #Unigram Encoding
    
    def uni_encoding(data, category):
        word_count = {}
        stopwords = nltk.corpus.stopwords.words('english')
        temp = data.loc[data['target'] == category]
        for form in tqdm(temp['full_text']):
            cleaned_form = re.sub(r'\W',' ', form)
            cleaned_form = re.sub(r'\s+',' ', cleaned_form)
            cleaned_form = re.sub(r'\d','', cleaned_form)
            cleaned_form = cleaned_form.lower()
            tokens = nltk.word_tokenize(cleaned_form)
            for token in tokens:
                if token in stopwords:
                    continue
                if token not in word_count.keys():                 
                    word_count[token] = 1
                else: 
                    word_count[token] += 1
        return word_count
    
    print()
    print('  => Tokenizing Data for 3 Classes...')
    print()
    
    up_dict = uni_encoding(X_train, 'UP')
    down_dict = uni_encoding(X_train, 'DOWN')
    stay_dict = uni_encoding(X_train, 'STAY')
    
    all_word_count = {**up_dict, **stay_dict, **down_dict}
    all_word_count = {key:val for key, val in all_word_count.items() if val > 10}
                
    #Compute PMI for each Class
    
    print()
    print('  => Computing PMI...')
    print()
                
    total_freq = sum(all_word_count.values())
    pmi_dict = {}
    for token in tqdm(all_word_count.keys()):
        p_x = all_word_count[token] / total_freq
        max_cond = []
        for i in [up_dict, down_dict, stay_dict]:
            if token in i.keys():
                temp_sum = sum(i.values())
                max_cond.append(i[token] / temp_sum)
            else:
                max_cond.append(0)
        pmi_dict[token] = np.log(np.mean(max_cond) / p_x)
        
    #Takes n Best Unigrams
    
    highest_pmi = heapq.nlargest(n_unigrams, pmi_dict, key = pmi_dict.get)

    print()
    print('  => Encoding Unigrams...')
    print()

    form_vectors = []
    for form in tqdm(merged_data['full_text']):
        cleaned_form = re.sub(r'\W',' ', form)
        cleaned_form = re.sub(r'\s+',' ', cleaned_form)
        cleaned_form = re.sub(r'\d','', cleaned_form)
        cleaned_form = cleaned_form.lower()
        tokens = nltk.word_tokenize(cleaned_form)
        temp = []
        for token in highest_pmi:
            if token in cleaned_form:                 
                temp.append(1)
            else: 
                temp.append(0)
        form_vectors.append(temp)

    merged_data['unigram_vec'] = form_vectors

    #Quality Phrase Encoding
    
    print()
    print('  => Encoding Quality Phrases...')
    print()

    quality_phrases = pd.read_csv(phrase_file, sep = '\t', header = None)

    def clean(text):
        return text.lower()

    quality_phrases['cleaned'] = quality_phrases[1].apply(clean)

    top_phrases = quality_phrases['cleaned'].loc[quality_phrases[0] > threshhold].copy()

    phrase_vectors = []
    for form in tqdm(merged_data['full_text']):
        cleaned_form = form.lower()
        temp = []
        for phrase in top_phrases:
            if phrase in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        phrase_vectors.append(temp)

    merged_data['phrase_vec'] = phrase_vectors
    
    print()
    print(' => Done feature_encoding!')
    print()
    
    return merged_data
