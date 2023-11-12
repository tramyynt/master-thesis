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
import liwc_alike
from liwc import Liwc
import ast
from scipy.signal import savgol_filter

HOME_DIR = "/home_remote"
fname = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model"))
fname2 = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model2"))
#load model
doc2vec_model = Doc2Vec.load(fname)
doc2vec_model2 = Doc2Vec.load(fname2)


#feature names
relevant_features_name ={'liwc': ['i', 'AverageLength', 'friend', 'sad', 'family', 'feel', 'health',
       'sexual', 'anx', 'body', 'bio', 'ppron', 'filler', 'shehe', 'adverb',
       'swear', 'humans', 'excl', 'assent', 'discrep', 'you', 'pronoun',
       'negemo', 'past'],
                        'liwc_alike': ['Anxiety', 'I', 'Sadness', 'AverageLength', 'Affective Processes',
       'Sexuality', 'Family', 'Friends', 'Fillers', 'Health', 'Feeling',
       'Humans', 'Biological Processes', 'Time', 'Body', 'Negative Emotions',
       'Social Processes', 'Perceptual Processes', 'Insight',
       'Cognitive Processes', 'Motion', 'Positive Emotions', 'Tentative',
       'Ppronouns']}

relevant_features_name_without_Length ={'liwc': ['i', 'friend', 'sad', 'family', 'feel', 'health',
       'sexual', 'anx', 'body', 'bio', 'ppron', 'filler', 'shehe', 'adverb',
       'swear', 'humans', 'excl', 'assent', 'discrep', 'you', 'pronoun',
       'negemo', 'past'],
                        'liwc_alike': ['Anxiety', 'I', 'Sadness', 'Affective Processes',
       'Sexuality', 'Family', 'Friends', 'Fillers', 'Health', 'Feeling',
       'Humans', 'Biological Processes', 'Time', 'Body', 'Negative Emotions',
       'Social Processes', 'Perceptual Processes', 'Insight',
       'Cognitive Processes', 'Motion', 'Positive Emotions', 'Tentative',
       'Ppronouns']}

#liwc_alike data
liwc2 = pd.read_excel('/home_remote/dic_avg100_annotated_official.xlsx')
liwc2['Terms'] = liwc2['Term'].apply(lambda x: ast.literal_eval(x))
result = dict(zip(liwc2['Category'], liwc2['Terms']))

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


#add AverageLength, NumOfWritings to the vector
def add_to_counter(counter, key, value):
  counter[key] = value
  return counter

#get features
def get_features(df,relevant_features_name, type):
    #print(df)
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name['liwc'] if i != 'AverageLength']):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name['liwc_alike'] if i != 'AverageLength']):
            if item not in output:
                output[item] = 0

    #print(output)
   # df['vector'] = [output]
    average_length = df['AverageLength'][0]
    output['AverageLength'] = average_length

    vector_df = pd.DataFrame([output])
    #print(vector_df)
    #vector_df_norm = (vector_df - vector_df.min()) / (vector_df.max() - vector_df.min())
    #vector_df_norm['Label'] = df['Label']
    vector_df['SubjectId'] = df['SubjectId']
    #vector_df_norm = vector_df_norm.fillna(0)
    #print(vector_df_norm)
    temp = vector_df[relevant_features_name[type]]
    temp_vector = temp.values[0]
    #print(temp_vector)
    #normailize temp_vector
    if temp_vector.max() != 0:
        X = (temp_vector - temp_vector.min()) / (temp_vector.max() - temp_vector.min())
    #print(X)
    else:
        print(vector_df['SubjectId'])
        return None
    return X.reshape(1, -1)

def extract_features_no_addition(df, relevant_features_name, type):
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name_without_Length['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name_without_Length['liwc_alike']]):
            if item not in output:
                output[item] = 0

    vector_df = pd.DataFrame([output])
    temp = vector_df[relevant_features_name[type]]
    temp_norm = temp.div(temp.sum(axis=1), axis=0)
    temp_norm = temp_norm.fillna(0)
    
    #vector_df_norm = (vector_df - vector_df.min()) / (vector_df.max() - vector_df.min())
    #temp_norm['TrainSubjectId'] = df['TrainSubjectId']
    X = temp_norm.values[0]
    return X.reshape(1, -1)

def extract_features_no_addition_mimx(df, relevant_features_name, type):
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name_without_Length['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name_without_Length['liwc_alike']]):
            if item not in output:
                output[item] = 0

    vector_df = pd.DataFrame([output])
    temp = vector_df[relevant_features_name[type]]
    temp_vector = temp.values[0]
    #print(temp_vector)
    #normailize temp_vector
    if temp_vector.max() != 0:
        X = (temp_vector - temp_vector.min()) / (temp_vector.max() - temp_vector.min())
    #print(X)
    else:
        print('There is a None value in the vector')
        return None
    return X.reshape(1, -1)
 


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
    
    elif type == 'liwc':
        X = get_features(df, relevant_features_name, 'liwc')
        #X= extract_features_no_addition_mimx(df, relevant_features_name_without_Length, 'liwc')
        #smoothing with Savitzky-Golay filter
        #X = savgol_filter(X, window_length=4, polyorder=3, deriv=2)
        #print(X)
    elif type == 'liwc_alike':
        X = get_features(df, relevant_features_name, 'liwc')
        #X = extract_features_no_addition_mimx(df, relevant_features_name_without_Length, 'liwc_alike')
        #X= savgol_filter(X, window_length=4, polyorder=3, deriv=2)
        #print(X)
    return X




    