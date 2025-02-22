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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import fasttext
import fasttext.util
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")

HOME_DIR = "/home_remote"
fname = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model"))
fname2 = get_tmpfile(os.path.join(HOME_DIR,"master_thesis/model_evaluation/my_doc2vec_model2"))
#load model
doc2vec_model = Doc2Vec.load(fname)
doc2vec_model2 = Doc2Vec.load(fname2)
pca = joblib.load(os.path.join(HOME_DIR, "pca.pkl"))
scaler_alike = joblib.load(os.path.join(HOME_DIR, "scaler_alike.pkl"))
scaler_liwc = joblib.load( os.path.join(HOME_DIR, "scaler_liwc.pkl"))
scaler_alike_10 = joblib.load(os.path.join(HOME_DIR, "scaler_alike_10.pkl"))
scaler_liwc_10 = joblib.load( os.path.join(HOME_DIR, "scaler_liwc_10.pkl"))
#load fasttext model
ft = fasttext.load_model('/home_remote/fastText/cc.en.300.bin')

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
relevant_features_name_23 = {'liwc': ['i', 'friend', 'sad', 'family', 'feel', 'health',
       'sexual', 'anx', 'body', 'bio', 'ppron', 'filler', 'shehe', 'adverb',
       'swear', 'humans', 'excl', 'assent', 'discrep', 'you', 'pronoun',
       'negemo', 'past'],
                        'liwc_alike': ['Anxiety', 'I', 'Sadness', 'Affective Processes',
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
features = {'liwc':['affect', 'filler', 'posemo', 'time', 'funct', 'article', 'cogmech',
       'cause', 'work', 'pronoun', 'ppron', 'shehe', 'social', 'verb',
       'present', 'insight', 'relig', 'tentat', 'preps', 'achieve', 'space',
       'relativ', 'auxverb', 'ipron', 'conj', 'excl', 'assent', 'percept',
       'feel', 'adverb', 'negate', 'inhib', 'we', 'incl', 'past', 'negemo',
       'anger', 'death', 'bio', 'ingest', 'humans', 'motion', 'future',
       'discrep', 'certain', 'AverageLength', 'NumOfWritings', 'see',
       'leisure', 'body', 'i', 'they', 'money', 'sad', 'quant', 'health',
       'you', 'anx', 'number', 'sexual', 'hear', 'nonfl', 'family', 'swear',
       'home', 'friend'],
            'liwc_alike': ['Present tense', 'Positive Emotions', 'Articles', 'Work', 'Pronouns',
       'Ppronouns', 'Shehe', 'Social Processes', 'Friends', 'Humans',
       'Prepositions', 'Cognitive Processes', 'Insight',
       'Perceptual Processes', 'ipron', 'auxverb', 'Conjunctions', 'Negations',
       'Exclusive', 'Causation', 'Relativity', 'Adverbs', 'Space', 'We',
       'Inclusive', 'Seeing', 'Negative Emotions', 'Motion', 'Discrepancy',
       'You', 'I', 'Time', 'AverageLength', 'NumOfWritings', 'Sexuality',
       'Tentative', 'Past tense', 'Biological Processes', 'Body', 'Health',
       'Future tense', 'Certainty', 'Nonfluencies', 'Leisure', 'Achievement',
       'Home', 'They', 'Numbers', 'Money', 'Anger', 'Religion', 'Assent',
       'Sadness', 'Feeling', 'Family', 'Death', 'Affective Processes',
       'Anxiety', 'Hearing', 'Inhibition', 'Fillers']}
features_15 = {'liwc':['i', 'adverb', 'excl', 'sad', 'health', 'anx', 'pronoun', 'ppron',
       'affect', 'conj', 'cogmech', 'tentat', 'negemo', 'verb', 'discrep'],
               'liwc_alike': ['I', 'Affective Processes', 'Anxiety', 'Sadness', 'Insight',
       'Social Processes', 'Tentative', 'Cognitive Processes', 'Feeling',
       'Negative Emotions', 'Health', 'Adverbs', 'Ppronouns', 'AverageLength',
       'Perceptual Processes']}

features_pca = {'liwc_alike':['auxverb', 'Cognitive Processes', 'Insight', 'Inclusive', 'Work',
       'Perceptual Processes', 'Assent', 'Pronouns', 'ipron', 'Causation',
       'Past tense', 'Seeing', 'Feeling', 'Articles', 'Certainty', 'Ppronouns',
       'I', 'Prepositions', 'Tentative', 'Exclusive', 'Affective Processes',
       'Negative Emotions', 'Achievement', 'Social Processes', 'Friends',
       'Time', 'Relativity', 'Motion', 'Present tense', 'Discrepancy',
       'Negations', 'Conjunctions', 'Shehe', 'Humans', 'They', 'Adverbs',
       'Space', 'We', 'Positive Emotions', 'Health', 'Anxiety', 'Anger',
       'Hearing', 'Numbers', 'Future tense', 'Family', 'Leisure', 'You',
       'Sexuality', 'Body', 'Inhibition', 'Home', 'Biological Processes',
       'Sadness', 'Nonfluencies', 'Death', 'Money', 'Religion', 'Fillers']}

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
 
def get_features_all(df, type):
    #print(df)
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in features['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in features['liwc_alike']]):
            if item not in output:
                output[item] = 0

    vector_df = pd.DataFrame([output])
    #vector_df = pd.DataFrame(df['vector'].tolist(), index=df.index)
    vector_df_norm = (vector_df - vector_df.min()) / (vector_df.max() - vector_df.min())
    #vector_df_norm['Label'] = df['Label']
    #vector_df_norm['TrainSubjectId'] = df['TrainSubjectId']
    vector_df_norm = vector_df_norm.fillna(0)

    X = vector_df_norm
    #print(X)
    return X

def get_features_15(df,features_15, type):
    #print(df)
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in features_15['liwc'] if i != 'AverageLength']):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in features_15['liwc_alike'] if i != 'AverageLength']):
            if item not in output:
                output[item] = 0

    average_length = df['AverageLength'][0]
    output['AverageLength'] = average_length

    vector_df = pd.DataFrame([output])
    temp = vector_df[features_15[type]]
    temp_norm = temp.div(temp.sum(axis=1), axis=0)
    temp_norm = temp_norm.fillna(0)
    X = temp_norm
    return X


def get_feature_withPCA(df,pca, features_pca,type):
        #print(df)
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in features_pca['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in features_pca['liwc_alike']]):
            if item not in output:
                output[item] = 0
    
    vector_df = pd.DataFrame(output, index=df.index)
    vector_df_norm = vector_df.div(vector_df.sum(axis=1), axis=0)
    vector_df_norm = vector_df_norm.fillna(0)
    X = pca.transform(vector_df_norm)
    return X

def get_ngrams(text, n):
  n_grams = ngrams(word_tokenize(text), n)
  return [ ' '.join(grams) for grams in n_grams]

def frequency_distribution(grams, word):
    ls = []
    for i in grams:
        count = 0
        for j in i:
            if j == word:
                count += 1
    ls.append(count)
    return ls

def get_features_crafted_23(df, relevant_features_name_23, type):
    hand_crafted = ['LWF', 'FOG', 'FRE', 'DCR', 'AVG_SEN', 'AVG_PER_WORD']
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
        #print(output)
        #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name_23['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name_23['liwc_alike']]):
            if item not in output:
                output[item] = 0
    
    bigram =[ get_ngrams(i, 2) for i in df['text']]
    unigram = [get_ngrams(i, 1) for i in df['text']]

    depression = frequency_distribution(bigram, 'my depression')
    anxiety = frequency_distribution(bigram, 'my anxiety')
    count_I = frequency_distribution(unigram, 'I')

    vector_df = pd.DataFrame([output])
    re = vector_df[relevant_features_name_23[type]]
    for i in hand_crafted:
        re[i] = df[i]
    re['Depression'] = depression
    re['My Anxiety'] = anxiety
    re['word_I'] = count_I
    X = re
    return X


def get_features_crafted_full(df, type):
    hand_crafted = [
        'POS', 'PRP', 'VBD', 'Length_Title', 'Month', 'Hour',
       'LWF', 'FRE', 'DCR', 'FOG', 'AVG_SEN', 'AVG_PER_WORD', 'My_Depression',
       'My_Anxiety', 'My_Therapist', 'word_I', 'word_I_title',
       'Diagnosed_Depression', 'Antidepressants', 'NumOfWritings']
    
    relevant_features_name23 ={'liwc': ['i', 'friend', 'sad', 'family', 'feel', 'health',
       'sexual', 'anx', 'body', 'bio', 'ppron', 'filler', 'shehe', 'adverb',
       'swear', 'humans', 'excl', 'assent', 'discrep', 'you', 'pronoun',
       'negemo', 'past'],
                        'liwc_alike': ['Anxiety', 'I', 'Sadness', 'Affective Processes',
       'Sexuality', 'Family', 'Friends', 'Fillers', 'Health', 'Feeling',
       'Humans', 'Biological Processes', 'Time', 'Body', 'Negative Emotions',
       'Social Processes', 'Perceptual Processes', 'Insight',
       'Cognitive Processes', 'Motion', 'Positive Emotions', 'Tentative',
       'Ppronouns']}
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
    #print(output)
    #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name23['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name23['liwc_alike']]):
            if item not in output:
                output[item] = 0
    vector_df = pd.DataFrame(output, index=df.index)
    #vector_df_norm = vector_df.div(vector_df.sum(axis=1), axis=0)
    #vector_df_norm['Label'] = df['Label']
    #vector_df_norm['TrainSubjectId'] = df['TrainSubjectId']
    vector_df= vector_df.fillna(0)
    #corr = vector_df_norm.corr()
    #corr_label = corr['Label'].sort_values(ascending=False)
    #relevant_features = corr_label[1:15]
    #relevant_features_name = relevant_features.index.values
    re = vector_df[relevant_features_name23[type]]
    for i in hand_crafted:
        re[i] = df[i]
    X = re
    return X


def get_features_crafted_10(df, type):
    hand_crafted = [
        'POS', 'PRP', 'VBD', 'Length_Title', 'Month', 'Hour',
       'LWF', 'FRE', 'DCR', 'FOG', 'AVG_SEN', 'AVG_PER_WORD', 'My_Depression',
       'My_Anxiety', 'My_Therapist', 'word_I', 'word_I_title',
       'Diagnosed_Depression', 'Antidepressants', 'NumOfWritings']
    
    relevant_features_name10 ={'liwc': ['i', 'friend', 'sad','sexual', 'anx','ppron', 'discrep', 'pronoun','negemo', 'past'],
                        'liwc_alike': ['Anxiety', 'I', 'Sadness', 'Negative Emotions','Social Processes', 'Insight','Cognitive Processes', 'Motion', 'Positive Emotions','Ppronouns']}
    if type == 'liwc':
        liwc = Liwc(os.path.join(HOME_DIR, "master_thesis/LIWC2007_English100131.dic"))
        output = liwc.parse(word_tokenize(df['text'][0]))
    #print(output)
    #relevant features for liwc except AverageLength
        for item in pd.Series([i for i in relevant_features_name10['liwc']]):
            if item not in output:
                output[item] = 0
    elif type == 'liwc_alike':
        output = liwc_alike.main(df['text'][0], result)
        for item in pd.Series([i for i in relevant_features_name10['liwc_alike']]):
            if item not in output:
                output[item] = 0
    vector_df = pd.DataFrame(output, index=df.index)
    #vector_df_norm = vector_df.div(vector_df.sum(axis=1), axis=0)
    #vector_df_norm['Label'] = df['Label']
    #vector_df_norm['TrainSubjectId'] = df['TrainSubjectId']
    vector_df= vector_df.fillna(0)
    #corr = vector_df_norm.corr()
    #corr_label = corr['Label'].sort_values(ascending=False)
    #relevant_features = corr_label[1:15]
    #relevant_features_name = relevant_features.index.values
    re = vector_df[relevant_features_name10[type]]
    for i in hand_crafted:
        re[i] = df[i]
    X = re
    return X

def get_documents_matrix(documents, max_words=100, embedding_dim=300, embedding_model=ft):
    document_matrices = []

    for document in documents:
        # Split document into words
        words = document.split()[:max_words]

        # Get word embeddings using FastText
        embeddings = [ft.get_word_vector(word) for word in words]

        document_matrices.append(embeddings)

    # Pad each sequence of embeddings to a common length
    padded_document_matrices = pad_sequences(document_matrices, maxlen=max_words, dtype='float32', padding='post', truncating='post')

    return padded_document_matrices


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
       # X = get_features_all(df, 'liwc')
        #X = get_features(df, relevant_features_name, 'liwc')
        #X= extract_features_no_addition_mimx(df, relevant_features_name_without_Length, 'liwc')
        #smoothing with Savitzky-Golay filter
        #X = savgol_filter(X, window_length=4, polyorder=3, deriv=2)
        #X = get_features_crafted_23(df, relevant_features_name_23, 'liwc')
        X1 = get_features_crafted_full(df,'liwc')
        X = scaler_liwc.transform(X1)

        #print(X)

    elif type == 'liwc_alike':

        #X = get_features_all(df, 'liwc_alike')
        #X = get_features(df, relevant_features_name, 'liwc_alike')
        #X = extract_features_no_addition_mimx(df, relevant_features_name_without_Length, 'liwc_alike')
        #X= savgol_filter(X, window_length=4, polyorder=3, deriv=2)
        #X = get_feature_withPCA(df,pca,features_pca, 'liwc_alike')
       # X = get_features_crafted_23(df, relevant_features_name_23, 'liwc_alike')
        X1 = get_features_crafted_full(df,'liwc_alike')
        X = scaler_alike.transform(X1)
        # X1 = get_features_crafted_10(df,'liwc_alike')
        # X = scaler_alike_10.transform(X1)
        #print(X)
    
    elif type == 'cnn':
        X= get_documents_matrix(df['text'])
    
    return X




    