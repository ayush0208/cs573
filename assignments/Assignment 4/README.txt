Name - Ayush Garg
PUID - 0033794803

Python version used for this is 3.9.1
Dependencies required - numpy, pandas, matplotlib, multiprocessing 

                                                Instructions to run code

For Q1 - for preprocessing the dataset
Command to be entered on terminal: python preprocess-assg4.py
Output: Provides two files trainingSet.csv and testSet.csv

For Q2 - for running Decision Trees
Command to be entered on terminal: python trees.py trainingSet.csv testSet.csv 1
Output: Training Accuracy DT: 0.77
        Testing Accuracy DT: 0.72

For Q2 - for running Bagging
Command to be entered on terminal: python trees.py trainingSet.csv testSet.csv 2
Output: Training Accuracy BT: 0.79
        Testing Accuracy BT: 0.75

For Q2 - for running Random Forest
Command to be entered on terminal: python trees.py trainingSet.csv testSet.csv 3
Output: Training Accuracy RF: 0.76
        Testing Accuracy RF: 0.73

For Q3 - for comparing accuracy with change in depth of the decision tree
Comparing models accuracy   (takes around  50(terminal)/ 34(jupyter notebook) minutes to run this program)

Command to be entered on terminal: python cv_depth.py
Output: Plot showing the accuracies for all the three models

For Q4 - for comparing accuracy with change in fraction of the dataset used
Comparing models accuracy   (takes around 38-40 (terminal)/ 16(jupyter notebook) minutes to run this program)

Command to be entered on terminal: python cv_frac.py
Output: Plot showing the accuracies for all the three models

For Q5 - for comparing accuracy with change in number of trees being used
Comparing models accuracy   (takes around 70(terminal) / 40(jupyter notebook) minutes to run this program)

Command to be entered on terminal: python cv_numtrees.py
Output: Plot showing the accuracies for all the three models


For Bonus Question
Command to be entered on terminal: python neuralNetwork.py
Training Accuracy NN: 0.73
Testing Accuracy NN: 0.72