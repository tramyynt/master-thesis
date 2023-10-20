import xml.etree.ElementTree as ET
import pandas as pd
import os

import random

def get_all_xml_files_in_a_folder(folder_path):
    xml_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))

    return xml_files

def extract_data_from_xml(xml_file):
    individual_writings = pd.DataFrame(columns = ['Title', 'Date', 'Text'])
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for writing in root.findall('WRITING'):
            writing_title = writing.find('TITLE').text
            writing_date = writing.find('DATE').text
            writing_text = writing.find('TEXT').text
            individual_writings = pd.concat([
                individual_writings,
                pd.DataFrame.from_dict({"Title": [writing_title], "Date": [writing_date], "Text": [writing_text] })
            ], ignore_index=True)

        subject_id = root.find('ID').text

        return (subject_id, individual_writings)
    except Exception as e:
        print(f"[ERROR] Can't parsing {xml_file}: {str(e)}")
        return None

def has_subject_id_been_predicted(subject_id, previous_predicted_results):
    if previous_predicted_results is None:
        return (False, None)

    previous_predicted_of_subject_series = previous_predicted_results.loc[previous_predicted_results['SubjectId'] == subject_id]
    if previous_predicted_of_subject_series.empty:
        return (False, None)

    previous_predicted_risk = previous_predicted_of_subject_series['Risk'].iloc[0]
    if previous_predicted_risk == 0:
        return (False, None)

    return (True, previous_predicted_risk)

def predict_from_chunk_data(model, path_to_nth_chunk_folder, previous_predicted_results = None):
    # Load the data
    predicted_results = pd.DataFrame(columns = ['SubjectId', 'Risk'])

    for xml_file in get_all_xml_files_in_a_folder(path_to_nth_chunk_folder):
        subject_id, individual_writings = extract_data_from_xml(xml_file)

        if individual_writings is None:
            continue

        has_predicted, previous_risk = has_subject_id_been_predicted(subject_id, previous_predicted_results)
        if has_predicted:
            print(f"Skip prediction of subject {subject_id} because it has been predicted")
            predicted_results = pd.concat([
                predicted_results,
                pd.DataFrame.from_dict({"SubjectId": [subject_id], "Risk": [previous_risk]})
            ], ignore_index=True)
            continue

        # Predict the risk of each user (0: skip this chunk; 1: positive; 2: negative)
        # TODO: option to join all documents or using one document at a time
        # risk = model.predict(individual_writings['Text'])
        risk = random.randint(0, 1)

        # if risk != 0:
        #     print(f"Predicted subject {subject_id} with risk {risk}")

        predicted_results = pd.concat([
            predicted_results,
            pd.DataFrame.from_dict({"SubjectId": [subject_id], "Risk": [risk]})
        ], ignore_index=True)

    return predicted_results.sort_values(by=['SubjectId'], ignore_index=True)

def write_predicted_results_to_file(predicted_results, nth_chunk, path_to_result_folder = "./results"):
    predicted_results.to_csv(f"{path_to_result_folder}/mynguyen_{nth_chunk}.txt", sep="\t", index=False, header=False)

print("******************************")
print("Start predicting chunk 1")
chunk_1_results = predict_from_chunk_data(None, "./test_data/chunk 1")
write_predicted_results_to_file(chunk_1_results, 1)

print("******************************")
print("Start predicting chunk 2")
chunk_2_results = predict_from_chunk_data(None, "./test_data/chunk 2", chunk_1_results)
write_predicted_results_to_file(chunk_2_results, 2)

print("******************************")
print("Start predicting chunk 3")
chunk_3_results = predict_from_chunk_data(None, "./test_data/chunk 3", chunk_2_results)
write_predicted_results_to_file(chunk_3_results, 3)

print("******************************")
print("Start predicting chunk 4")
chunk_4_results = predict_from_chunk_data(None, "./test_data/chunk 4", chunk_3_results)
write_predicted_results_to_file(chunk_4_results, 4)

print("******************************")
print("Start predicting chunk 5")
chunk_5_results = predict_from_chunk_data(None, "./test_data/chunk 5", chunk_4_results)
write_predicted_results_to_file(chunk_5_results, 5)

print("******************************")
print("Start predicting chunk 6")
chunk_6_results = predict_from_chunk_data(None, "./test_data/chunk 6", chunk_5_results)
write_predicted_results_to_file(chunk_6_results, 6)

print("******************************")
print("Start predicting chunk 7")
chunk_7_results = predict_from_chunk_data(None, "./test_data/chunk 7", chunk_6_results)
write_predicted_results_to_file(chunk_7_results, 7)

print("******************************")
print("Start predicting chunk 8")
chunk_8_results = predict_from_chunk_data(None, "./test_data/chunk 8", chunk_7_results)
write_predicted_results_to_file(chunk_8_results, 8)

print("******************************")
print("Start predicting chunk 19")
chunk_9_results = predict_from_chunk_data(None, "./test_data/chunk 9", chunk_8_results)
write_predicted_results_to_file(chunk_9_results, 9)

print("******************************")
print("Start predicting chunk 10")
chunk_10_results = predict_from_chunk_data(None, "./test_data/chunk 10", chunk_9_results)
write_predicted_results_to_file(chunk_10_results, 10)
