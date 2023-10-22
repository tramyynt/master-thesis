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


def read_corpus(text, tokens_only=False):
    tokens = gensim.utils.simple_preprocess(text)
    if tokens_only:
        yield tokens

def pre_processing(text, type):
    if type == 'tfidf':
        text_clean = text.apply(clean_text)
        tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7)
        X = tfidfconverter.fit_transform(text_clean).toarray()

    elif type == 'doc2vec':
        tokens = read_corpus(text)
        X= np.concatenate((doc2vec_model.infer_vector(tokens), doc2vec_model2.infer_vector(tokens)), axis=None)
    return X.reshape(1, -1)


    