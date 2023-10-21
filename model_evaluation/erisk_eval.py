import pandas as pd
import numpy as np
import argparse

# To calculate the metrics of the classification algorithm
def erde_evaluation(goldenTruth_path, algorithmResult_path, o, o_percentage):
    # Transform files into tables
    try:
        data_golden = pd.read_csv(goldenTruth_path, sep=" ", header=None, names=['subj_id', 'true_risk'])

        data_result = pd.read_csv(algorithmResult_path, sep=" ", header=None, names=['subj_id', 'risk_decision', 'delay', 'total_writings'])

        # Merge tables (data) on the common field 'subj_id' to compare the true risk and the decision risk
        merged_data = data_golden.merge(data_result, on='subj_id', how='left')

        # Add a column to store the individual ERDE of each user
        merged_data.insert(loc=len(merged_data.columns), column='erde', value=1.0)

        # Add a column to store the individual ERDE_percentage of each user
        merged_data.insert(loc=len(merged_data.columns), column='erde_percentage', value=1.0)

        # Variables
        risk_d = merged_data['risk_decision']
        t_risk = merged_data['true_risk']
        k = merged_data['delay']
        total_writings = merged_data['total_writings']
        erde = merged_data['erde']
        erde_percentage = merged_data['erde_percentage']

        # Count of how many true positives there are
        true_pos = len(merged_data[t_risk == 1])

        # Count of how many positive cases the system decided there were
        pos_decisions = len(merged_data[risk_d == 1])

        # Count of how many of them are actually true positive cases
        pos_hits = len(merged_data[(t_risk == 1) & (risk_d == 1)])

        # Total count of users
        total_users = len(merged_data)

        # ERDE calculus
        for i in range(total_users):
            if risk_d[i] == 1 and t_risk[i] == 0: # false positive
                c_fp = float(true_pos) / total_users
                erde.iat[i] = c_fp
                erde_percentage.iat[i] = c_fp
            elif risk_d[i] == 0 and t_risk[i] == 1: # false negative
                erde.iat[i] = 1.0
                erde_percentage.iat[i] = 1.0
            elif risk_d[i] == 1 and t_risk[i] == 1: # true positive
                erde.iat[i] = 1.0 - (1.0 / (1.0 + np.exp(k[i] - o)))
                p = 100.0 * k[i] / total_writings[i]
                erde_percentage.iat[i] = 1.0 - (1.0 / (1.0 + np.exp(p - o_percentage)))
            elif risk_d[i] == 0 and t_risk[i] == 0: # true negative
                erde.iat[i] = 0.0
                erde_percentage.iat[i] = 0.0

        # Calculus of F1, Precision, Recall, and global ERDE
        precision = float(pos_hits) / pos_decisions if pos_decisions != 0 else 0
        recall = float(pos_hits) / true_pos if true_pos != 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        erde_global = erde.mean() * 100
        erde_percentage_global = erde_percentage.mean() * 100

        indiv_erde = merged_data.loc[:, ['subj_id', 'erde', 'erde_percentage']]
        print(indiv_erde.to_string(index=False))
        print(f'Global ERDE (with o = {o}): {erde_global:.2f}%')
        print(f'Global ERDE percentage (with o = {o_percentage}): {erde_percentage_global:.2f}%')
        print(f'F1: {F1:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')

    except Exception as e:
        print(f'Some file or directory doesn\'t exist or an error has occurred: {str(e)}')

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('-gpath', help='(Obligatory) path to golden truth file.', required=True, nargs=1, dest="gpath")
parser.add_argument('-ppath', help='(Obligatory) path to prediction file from a system.', required=True, nargs=1, dest="ppath")
parser.add_argument('-o', help='(Obligatory) o parameter.', required=True, nargs=1, dest="o")
parser.add_argument('-oper', help='(Obligatory) o percentage parameter.', required=True, nargs=1, dest="oper")

args = parser.parse_args()

gpath = args.gpath[0]
ppath = args.ppath[0]
o = int(args.o[0])
o_percentage = int(args.oper[0])


erde_evaluation(gpath, ppath, o, o_percentage)
