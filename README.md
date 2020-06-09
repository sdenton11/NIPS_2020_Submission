# SVM Actionability NIPS_2020_Submission
In this directory is the code for SVM Actionability NIPS 2020 Submission.

### Requirements

To satisfy necessary requirements run `pip install -r requirements.txt`

### Algorithm Code

The file `SVM_Actionability.py` has a main function that will run three main helper functions:

1. `run_linear_svm()` -- This method will create and run the solution to the linear SVM for a toy example. There are three optional arguments:
  - the weights
  - the directory for results
  - the number of points

2. `run_nonlinear_svm()` -- This method will create and run the solution to the non-linear SVM for a toy example. There are five optional arguments:
  - the weights
  - the directory for results
  - the number of points
  - the type of data (moon or circle)
  - the type of svm (rbf or poly)

3. `run_atherosclerosis_data()` -- This method will create and run an SVM to fit the Atherosclerosis data problem. There are three optional arguments:
  - the weights for the features ['SUBSC', 'TRIC', 'TRIGL', 'SYST', 'DIAST', 'BMI', 'WEIGHT', 'CHLST', 'ALCO_CONS', 'TOBA_CONSO']
  - the directory for results
  - the type of svm (rbf or poly).

Note: all the saving of the figures and data has been commented out for convenience reasons. To save information on the results, uncomment the desired sections.
