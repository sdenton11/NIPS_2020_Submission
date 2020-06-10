# SVM Actionability NIPS_2020_Submission
In this directory is the code for SVM Actionability NIPS 2020 Submission.

## Requirements

To satisfy necessary requirements run `pip install -r requirements.txt`

## Algorithm Code

The file `SVM_Actionability.py` has a main function that will run three main commands:

1. `run_linear_svm()` -- This method will create and run the solution to the linear SVM for a toy example. There are five optional arguments:
  - the weights
  - the directory for results
  - the number of points
  - the summary of the results
  - graph each example point

2. `run_nonlinear_svm()` -- This method will create and run the solution to the non-linear SVM for a toy example. There are seven optional arguments:
  - the weights
  - the directory for results
  - the number of points
  - the type of data (moon or circle)
  - the type of svm (rbf or poly)
  - the summary of the results
  - graph each example point

3. `run_atherosclerosis_data()` -- This method will create and run an SVM to fit the Atherosclerosis data problem. There are four optional arguments:
  - the weights for the features ['SUBSC', 'TRIC', 'TRIGL', 'SYST', 'DIAST', 'BMI', 'WEIGHT', 'CHLST', 'ALCO_CONS', 'TOBA_CONSO']
  - the directory for results
  - the type of svm (rbf or poly)
  - the summary of the results
  
### Running the Commands
To run the commands, you can specify which command as well as the optional arguments via the command-line. The usage is shown below:
```
usage: SVM_Actionability.py [-h] [-w WEIGHTS [WEIGHTS ...]] [-r RESULT_DIR]
                            [-n NUM_POINTS] [-d {circle,moon}]
                            [-svm {rbf,poly}] [-S] [-g]
                            C

Perform SVM Actionability.

positional arguments:
  C                     the type of SVM action to perform ('linear', 'non-
                        linear', or 'atheroslcerosis').

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS [WEIGHTS ...], --weights WEIGHTS [WEIGHTS ...]
                        the weights (between 0 and 1) for the associated model
                        (length of 2 for linear/non-linear, length of 10 for
                        atherosclerosis).
  -r RESULT_DIR, --result_dir RESULT_DIR
                        the directory to save the results to (default='')
  -n NUM_POINTS, --num_points NUM_POINTS
                        the number of points to simulate (default: 100)
  -d {circle,moon}, --data_type {circle,moon}
                        the type of data for the non-linear SVM.
  -svm {rbf,poly}, --svm_type {rbf,poly}
                        the type of SVM for the non-linear SVM or
                        atherosclerosis SVM.
  -S, --summary         specify whether to save the summary of the results.
  -g, --graph_examples  specify whether to graph each example point.
```

Here are examples for each of the three commands:
**linear**: `python3 SVM_Actionability.py linear -w 1 0.5 -r results/ -n 123 -S -g`

**non-linear**: `python3 SVM_Actionability.py non-linear -w 1 0.2 -r results/ -n 123 -d circle -svm poly -S -g`

**atherosclerosis**: `python3 SVM_Actionability.py atherosclerosis -w 1 1 0.01 1 0.75 0.3 1 0.8 0.01 0.2 -r results/ -svm rbf -S`
