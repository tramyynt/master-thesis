import pandas as pd
import nltk
import os
import re
import numpy as np
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile

HOME_DIR = "/home_remote"
fname = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model"))
fname2 = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model2"))
#load model
doc2vec_model = Doc2Vec.load(fname)
doc2vec_model2 = Doc2Vec.load(fname2)

def clean_text(text):
    # lower text
    text = text.lower()
     #text = nltk.word_tokenize(text)
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all text
    text = ' '.join(text)
    return(text)


def read_corpus(df, tokens_only=False):
    #print(df)
    for i, line in enumerate(df['text']):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def avg_feature_vector(train_corpus):
    #df['Tag'] = train_corpus
    #get tags of train_corpus
    tags = [x.tags[0] for x in train_corpus]
    #print(tags)
    vector= [np.concatenate((doc2vec_model.infer_vector(train_corpus[x].words), doc2vec_model2.infer_vector(train_corpus[x].words)), axis=None) for x in tags]
    #get just 1 an averaged vector from df['Vector']
    return np.mean(vector, axis=0)

def pre_processing(df, type):
    if type == 'tfidf':
        text_clean = df['text'].apply(lambda x: clean_text(x))
        #load tfidf model
        tfidfconverter = joblib.load(os.path.join(HOME_DIR, "tfidf_model.pkl"))
        X_array = tfidfconverter.transform(text_clean).toarray()
        X = np.mean(X_array, axis=0).reshape(1, -1)


    elif type == 'doc2vec':
        corpus = list(read_corpus(df))
        #print(corpus)
        X = avg_feature_vector(corpus)
        #tokens = read_corpus(text)
        #X= np.concatenate((doc2vec_model.infer_vector(tokens), doc2vec_model2.infer_vector(tokens)), axis=None)
        X = X.reshape(1, -1)
    return X


    