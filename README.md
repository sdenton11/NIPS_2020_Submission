# SVM Actionability NIPS_2020_Submission
In this directory is the code for SVM Actionability NIPS 2020 Submission.

The file `SVM_Actionability.py` has a main function that will run three main helper functions:

1. `run_linear_svm()` -- This method will create and run the solution to the linear SVM for a toy example.

2. `run_nonlinear_svm()` -- This method will create and run the solution to the non-linear SVM for a toy example. If you would like to use the RBF kernel vs. the polynomial kernel, you have to change the SVM in use to be the corresponding svm. Additionally, in `create_nonlinear_data()` you can specify the type (moon or circle).

3. `run_atherosclerosis_data()` -- This method will create and run an SVM to fit the Atherosclerosis data problem. Again, to change between RBF kernel and polynomial kernel, you must change the SVM in use. 

Note: all the saving of the figures and data has been commented out for convenience reasons. To save information on the results, uncomment the desired sections.
