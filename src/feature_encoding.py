import re
import heapq
import nltk
import numpy as np
import pandas as pd

def text_encode(data_file, phrase_file, n_unigrams, threshhold, train_split, test_split, **kwargs):

    merged_data = pd.read_csv(data_file)
    
    #Cleaned Event Feature

    def event_clean(text):
        result = re.sub('\n', '', text)
        result = re.sub('\t+', '\t', result)
        result = re.split('\t', result)
        if len(result) > 0:
            return str.lower(result[1]).split(";")
        else:
            return 'Missing'

    cleaned_event = merged_data['event_type'].apply(event_clean)
    merged_data.insert(3, 'cleaned_event', cleaned_event)
    
    #Train, Val, Test Split for Encoding

    X_train, X_test = train_test_split(merged_data, test_size = 1 - train_split, random_state = 42)
    X_val, X_test = train_test_split(X_test, test_size = test_split / (1 - train_split), random_state = 42)
    
    #Unigram Encoding
    
    print(' => Tokenizing Data...')
    word_count = {}
    stopwords = nltk.corpus.stopwords.words('english')
    for form in X_train['full_text']:
        cleaned_form = re.sub(r'\W',' ', form)
        cleaned_form = re.sub(r'\s+',' ', cleaned_form)
        cleaned_form = cleaned_form.lower()
        tokens = nltk.word_tokenize(cleaned_form)
        for token in tokens:
            if token in stopwords:
                continue
            if token not in word_count.keys():
                word_count[token] = 1
            else:
                word_count[token] += 1

    #Takes n largest unigrams

    print(' => Encoding Unigrams...')
    most_freq = heapq.nlargest(n_unigrams, word_count, key=word_count.get)

    form_vectors = []
    for form in merged_data['full_text']:
        cleaned_form = re.sub(r'\W',' ', form)
        cleaned_form = re.sub(r'\s+',' ', cleaned_form)
        cleaned_form = cleaned_form.lower()
        tokens = nltk.word_tokenize(cleaned_form)
        temp = []
        for token in most_freq:
            if token in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        form_vectors.append(temp)

    merged_data['unigram_vec'] = form_vectors

    #Quality Phrase Encoding

    print(' => Encoding Quality Phrases...')
    quality_phrases = pd.read_csv(phrase_file, sep = '\t', header = None)

    def clean(text):
        return text.lower()

    quality_phrases['cleaned'] = quality_phrases[1].apply(clean)

    top_phrases = quality_phrases['cleaned'].loc[quality_phrases[0] > threshhold].copy()

    phrase_vectors = []
    for form in merged_data['full_text']:
        cleaned_form = cleaned_form.lower()
        temp = []
        for phrase in top_phrases:
            if phrase in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        phrase_vectors.append(temp)

    merged_data['phrase_vec'] = phrase_vectors

    #Exports to .pkl file for models to use

    print(' => Exporting to pkl...')
    merged_data.to_pickle(kwargs['out_dir'] + 'feature_encoded_merged_data.pkl')

    return
