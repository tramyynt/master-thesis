## Before running evaluation scripts

Need to format test files before running evaluation.

-   `writings_all_test_users.txt` needs to be separated by comma (",") and all spaces around are being trimmed.
-   `test_golden_truth.txt` needs to be separated by 1 space, not tab and all spaces around are being trimmed.

Need to uncomment the processing model in utils.py corresponding to the model used in predict.results.py


## Run evaluation

0. `cd model_evaluation`
1. `python3 predict_results.py`
2. `python3 aggregate_results.py -path ./results -wsource ./test_data/writings_all_test_users.txt`
3. `python3 erisk_eval.py -gpath ./test_data/test_golden_truth.txt -ppath ./results/mynguyen_global.txt -o 5 -oper 20`

## Other scripts
a. **Run lexicon_based models**
    
    1. `python3 predict_results_lexicon.py`
    
    2. `python3 aggregate_results.py -path ./results_lexicon -wsource ./test_data/writings_all_test_users.txt`
    
    3. `python3 erisk_eval.py -gpath ./test_data/test_golden_truth.txt -ppath ./results_lexicon/mynguyen_global.txt -o 5 -oper 20`

    
b. **Run neural networks models**

    1. `python3 predict_results_cnn.py`
    
    2. `python3 aggregate_results.py -path ./results_lexicon -wsource ./test_data/writings_all_test_users.txt`
    
    3. `python3 erisk_eval.py -gpath ./test_data/test_golden_truth.txt -ppath ./results_lexicon/mynguyen_global.txt -o 5 -oper 20`

    
c. **Run combined models**
    
    1. `python3 predict_results_combined.py`
    
    2. `python3 aggregate_results.py -path ./results_lexicon -wsource ./test_data/writings_all_test_users.txt`
    
    3. `python3 erisk_eval.py -gpath ./test_data/test_golden_truth.txt -ppath ./results_lexicon/mynguyen_global.txt -o 5 -oper 20`


