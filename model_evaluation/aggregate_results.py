import os
import xml.etree.ElementTree as et
import argparse

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('-path', help='(Obligatory) Path to folder where predicted results txt files are placed.', required=True, nargs=1, dest="path")
parser.add_argument('-wsource', help='(Obligatory) Source file of the number of writings.', required=True, nargs=1, dest="wsource")
args = parser.parse_args()

RESULT_FOLDER_PATH = args.path[0]
NUM_OF_WRITINGS_SOURCE = args.wsource[0]
CHUNKS = 10
RESULT_PREFIX = "mynguyen"

# Construct a dictionary with the number of writings of each user/subject
user_writings_dict = {}
user_writings = open(NUM_OF_WRITINGS_SOURCE, "r")
lines = user_writings.readlines()

for line in lines:
    id, number_of_writings = line.strip().replace(" ", "").split(",")

    # Each user has a list of 3 elements: number of writings, risk decision, and chunk number gave the predicted result
    user_writings_dict[id] = [int(number_of_writings), 0, 0]

user_writings.close()

# Read risk decision from each chunk
for i in range(1, CHUNKS + 1):

    # For chunk i
    f_user = open(os.path.join(RESULT_FOLDER_PATH, f"{RESULT_PREFIX}_{str(i)}.txt"), "r")
    lines = f_user.readlines()

    # Iterate over the records in the file
    for line in lines:
        subject_id, risk_decision = line.replace(" ", "").split(",")
        subject = user_writings_dict[subject_id]

        # if predicted result is 0, update the predicted result of user
        if int(subject[1]) == 0:
            subject[1] = int(risk_decision)
            subject[2] = i
    f_user.close()

# Write the final result to a file
final_result_file_path = os.path.join(RESULT_FOLDER_PATH, f"{RESULT_PREFIX}_global.txt")
final = open(final_result_file_path, "w")
for subject_id, subject in user_writings_dict.items():
    total_number_of_writings = subject[0]
    risk_decision = subject[1]
    chunk_number_gave_the_predicted_result = subject[2]

    if chunk_number_gave_the_predicted_result == CHUNKS:
        num_w = total_number_of_writings
    else:
        num_w = (total_number_of_writings // CHUNKS) * chunk_number_gave_the_predicted_result


    if int(risk_decision) == 2:
        risk_decision = 0

    final.write(subject_id + " " + str(risk_decision) + " " + str(num_w) + "\n")
final.close()

print(f"Results saved in output file: {final_result_file_path}")
