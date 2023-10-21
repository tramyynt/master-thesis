## Before running evaluation scripts

Need to format test files before running evaluation.

-   `writings_all_test_users.txt` need to be separated by comma (",") and all spaces around are being trimmed.
-   `test_golden_truth.txt` need to be separated by 1 space, not tab and all spaces around are being trimmed.

## Run evaluation

0. `cd model_evaluation`
1. `python3 predict_results.py`
2. `python3 aggregate_results.py -path ./results -wsource ./test_data/writings_all_test_users.txt`
3. `python3 erisk_eval.py -gpath ./test_data/test_golden_truth.txt -ppath ./results/mynguyen_global.txt -o 5`
