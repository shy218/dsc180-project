import re
import heapq
import numpy as np
import pandas as pd
import os

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def text_encode(data_file, phrase_file, n_unigrams, treshhold, data_dir):

    #Unigram Encoding

    word_count = {}
    stopwords = nltk.corpus.stopwords.words('english')

    for form in data_file['full_text']:
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

    most_freq = heapq.nlargest(n_unigrams, word_count, key=word_count.get)

    form_vectors = []
    for form in data_file['full_text']:
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

    data_file['unigram_vec'] = form_vectors

    #Quality Phrase Encoding

    quality_phrases = pd.read_csv(phrase_file, sep = '\t', header = None)

    def clean(text):
        return text.lower()

    quality_phrases['cleaned'] = quality_phrases[1].apply(clean)

    top_phrases = quality_phrases['cleaned'].loc[quality_phrases[0] > treshhold].copy()

    phrase_vectors = []
    for form in data_file['full_text']:
        cleaned_form = cleaned_form.lower()
        temp = []
        for phrase in top_phrases:
            if phrase in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        phrase_vectors.append(temp)

    data_file['phrase_vec'] = phrase_vectors

    #Exports to .pkl file for models to use
    # data_file.to_pickle('feature_encoded_data_file.pkl')
    os.system('mkdir '+ data_dir +'tmp/')
    data_file.to_pickle(data_dir + 'tmp/feature_encoded_data_file.pkl')

    return
