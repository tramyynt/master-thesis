import xml.etree.ElementTree as ET
import pandas as pd
import os
import pandas as pd
import nltk
import os
import re
import numpy as np
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import joblib
import utils
import subprocess
import string
pd.options.mode.chained_assignment = None

HOME_DIR = "/home_remote"

# Specify the path to the Python script you want to run
script_path = os.path.join(HOME_DIR, 'master_thesis/model_evaluation/utils.py')

# Run the script
subprocess.call(['python', script_path])

#read model joblib
tfidf = joblib.load(os.path.join(HOME_DIR, "lg1.pkl"))
#doc2vec = joblib.load(os.path.join(HOME_DIR, "conventional_model.pkl"))
doc2vec = joblib.load(os.path.join(HOME_DIR, "lg3.pkl"))

# ------------------------- FUNCTIONS ------------------------- #
def get_all_xml_files_in_a_folder(folder_path):
    xml_files = []

    for entry in os.scandir(folder_path):
        if entry.is_file() and entry.name.endswith('.xml'):
            xml_files.append(entry.path)

    return xml_files

def extract_data_from_xml(xml_file, nth_chunk):
    individual_writings = pd.DataFrame(columns = ['SubjectId', 'Title', 'Text', 'Chunk', 'DataPath'])
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        subject_id = root.find('ID').text

        for writing in root.findall('WRITING'):
            title_tag = writing.find('TITLE')
            if title_tag is None:
                writing_title = ""
            else:
                writing_title = title_tag.text.strip()
            writing_text = writing.find('TEXT').text.strip()
            individual_writings = pd.concat([
                individual_writings,
                pd.DataFrame.from_dict({ "SubjectId": [subject_id], "Title": [writing_title], "Text": [writing_text], "Chunk": [nth_chunk], "DataPath": [xml_file] })
            ], ignore_index=True)

        # individual_writings['TitleAndText'] = individual_writings['Title'] + individual_writings['Text']
        return individual_writings
    except Exception as e:
        print(f"[ERROR] Can't parsing {xml_file}: {str(e)}")
        return None



def has_subject_id_been_predicted(subject_id, previous_predicted_results):
    if previous_predicted_results is None:
        return (False, None)

    previous_predicted_of_subject_series = previous_predicted_results.loc[previous_predicted_results['SubjectId'] == subject_id]
    if previous_predicted_of_subject_series.empty:
        return (False, None)
    #if that user is predicted as O -> will be predicted in next chunk
    previous_predicted_risk = previous_predicted_of_subject_series['Risk'].iloc[0]
    if previous_predicted_risk == 0:
        return (False, None)

    return (True, previous_predicted_risk)

def predict_from_chunk_data(model, type, all_writings, all_users, previous_predicted_results = None):
    """
    :param model: the model used for prediction
    :param type: the type of model (tfidf, doc2vec)
    :param all_writings all writings of all users from current chunk and all previous chunks
    :param all_users list of subject/user id
    """
    
    predicted_results = pd.DataFrame(columns = ['SubjectId', 'Risk'])

    for subject_id in all_users:
        has_predicted, previous_risk = has_subject_id_been_predicted(subject_id, previous_predicted_results)
        if has_predicted:
            print(f"Skip prediction of subject {subject_id} because it has been predicted")
            predicted_results = pd.concat([
                predicted_results,
                pd.DataFrame.from_dict({"SubjectId": [subject_id], "Risk": [previous_risk]})
            ], ignore_index=True)
            continue

        all_writings_of_subject = all_writings.loc[all_writings['SubjectId'] == subject_id]
        if all_writings_of_subject.empty:
            print(f"[ERROR] Subject {subject_id} has no writings")
            continue

        # Predict the risk of each user (0: skip the chunk; 1: positive; 2: negative)
        # TODO: option to join all documents or using one document at a time

        #join all text in each individual_writings
        all_writings_of_subject['text'] = all_writings_of_subject['Title'] + all_writings_of_subject['Text']
        #text = title_and_text.str.cat(sep=' '))
        #print(all_writings_of_subject)
        data = utils.pre_processing(all_writings_of_subject, type)
        #print(data)
        #risk = model.predict(data)
        prob = model.predict_proba(data)
        if type == 'doc2vec':
            if prob[0,1] > 0.7:
                risk = 1
            elif prob[0,1] > 0.6 and all_writings_of_subject.shape[0] > 10:
                risk = 1
            elif prob[0,1] > 0.5 and all_writings_of_subject.shape[0] > 15:
                risk = 1
            elif prob[0,1] < 0.05:
                risk = 2
            elif prob[0,1] < 0.1 and all_writings_of_subject.shape[0] > 10:
                risk = 2
            elif prob[0,1] < 0.15 and all_writings_of_subject.shape[0] > 20:
                risk = 2
            else:
                risk = 0
        else:
            if prob[0,1] > 0.3:
                risk = 1
            #elif prob[0,1] > 0.6 and all_writings_of_subject.shape[0] > 10:
                #risk = 1
            elif prob[0,1] > 0.25 and all_writings_of_subject.shape[0] > 10:
                risk = 1
            elif prob[0,1] < 0.05:
                risk = 2
            elif prob[0,1] < 0.1 and all_writings_of_subject.shape[0] > 10:
                risk = 2
            elif prob[0,1] < 0.15 and all_writings_of_subject.shape[0] > 20:
                risk = 2
            else:
                risk = 0
        # risk = random.randint(0, 1)

        # if risk != 0:
        #     print(f"Predicted subject {subject_id} with risk {risk}")

        predicted_results = pd.concat([
            predicted_results,
            pd.DataFrame.from_dict({"SubjectId": [subject_id], "Risk": risk})
        ], ignore_index=True)

    return predicted_results.sort_values(by=['SubjectId'], ignore_index=True)

def read_chunk_data(path_to_nth_chunk_folder, nth_chunk):
    """
    Read all data from a chunk folder and return dataframe of all writings.
    Each row of the dataframe is one writing (comment) of one user.

    :param path_to_nth_chunk_folder: path to the folder of the chunk
    :param nth_chunk: the number of the chunk

    :return all_writings: a dataframe of all writings of nth chunk
    """
    all_writings = pd.DataFrame()
    for xml_file in get_all_xml_files_in_a_folder(path_to_nth_chunk_folder):
        individual_writings = extract_data_from_xml(xml_file, nth_chunk=nth_chunk)

        if individual_writings is None:
            continue

        all_writings = pd.concat([
            all_writings,
            individual_writings,
        ], ignore_index=True)

    return all_writings

def get_list_of_subject_id(path_to_writings_all_test_users):
    df = pd.read_csv(path_to_writings_all_test_users, sep=',', header=None)
    return df.iloc[:, 0]

def write_predicted_results_to_file(predicted_results, nth_chunk, path_to_result_folder = os.path.join(HOME_DIR, "master_thesis/model_evaluation/results")):
    predicted_results.to_csv(f"{path_to_result_folder}/mynguyen_{nth_chunk}.txt", sep=",", index=False, header=False)


# ------------------------- MAIN ------------------------- #
path_to_writings_all_test_users = os.path.join(HOME_DIR, 'eRisk2018training/2017_test/writings_all_test_users.txt')
# path_to_writings_all_test_users = os.path.join(HOME_DIR, 'master_thesis/model_evaluation/test_data/writings_all_test_users.txt')
all_users = get_list_of_subject_id(path_to_writings_all_test_users)
all_writings = pd.DataFrame()

previous_predicted_results = None
# loop from 1 to 10
for chunk_i in range(1, 11):
    print("--------------------------------------------------")
    chunk_path = os.path.join(HOME_DIR, f"eRisk2018training/2017_test/chunk {chunk_i}")
    # chunk_path = os.path.join(HOME_DIR, f"master_thesis/model_evaluation/test_data/chunk {chunk_i}")
    chunk_writings = read_chunk_data(chunk_path, nth_chunk=chunk_i)
    all_writings = pd.concat([all_writings, chunk_writings], ignore_index=True)

    print(f"Start predicting chunk {chunk_i}")
    predicted_results = predict_from_chunk_data(doc2vec, 'doc2vec', all_writings=all_writings, all_users=all_users, previous_predicted_results=previous_predicted_results)

    if (chunk_i == 10):
        predicted_results.loc[predicted_results["Risk"] == 0, "Risk"] = 2

    write_predicted_results_to_file(predicted_results, chunk_i)

    previous_predicted_results = predicted_results
