Name - Ayush Garg
PUID - 0033794803

Python version used for this is 3.9.1
Dependencies required - numpy, pandas, matplotlib

                                                Instructions to run code

For Q1
Command to be entered on terminal: python preprocess-assign3.py
Output: Mapped vector for female in column gender:  [1]
        Mapped vector for Black/African American in column race:  [0, 1, 0, 0]
        Mapped vector for Other in column race_o:  [0, 0, 0, 0]
        Mapped vector for economics in column field:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

For Q2 - for running Logistic Regression
Command to be entered on terminal: python lr_svm.py trainingSet.csv testSet.csv 1
Output: Training Accuracy LR: 0.65
        Testing Accuracy LR: 0.65

For Q2 - fur running SVM
Command to be entered on terminal: python lr_svm.py trainingSet.csv testSet.csv 2
Output: Training Accuracy SVM: 0.56
        Testing Accuracy SVM: 0.55

For Q3

a) Preprocessing dataset for NBC
Command to be entered on terminal: python preprocessNBC.py
Output:    Training and testing files are generated

b) comparing models accuracy   (takes around 3-4 minutes to run this program)
Command to be entered on terminal: python cv.py
Output: Plot showing the accuracies for all the three models