#/********************************************************************************/#
#/*     Objective : Classification Audit Tracker                                  /#                           */                           */
#/*                                                                               /#
#/*     Prepared By : Rajendra Mehta                                              /#
#/*                                                                               /#  
#/********************************************************************************/#

import tkinter as Tkinter

if __name__ == '__main__':
    form = Tkinter.Tk()

    getFld = Tkinter.IntVar()

    form.wm_title('Classification Audit Tracker_Rajednra_Mehta')

    stepOne = Tkinter.LabelFrame(form, text=" 1. Enter File Details: ")
    stepOne.grid(row=0, columnspan=7, sticky='W', \
                 padx=5, pady=5, ipadx=5, ipady=5)

    helpLf = Tkinter.LabelFrame(form, text=" Quick Help ")
    helpLf.grid(row=0, column=9, columnspan=2, rowspan=8, \
                sticky='NS', padx=5, pady=5)
    helpLbl = Tkinter.Label(helpLf, text="Help will come - Ask To Tina.")
    helpLbl.grid(row=0)


    inFileLbl = Tkinter.Label(stepOne, text="Select the Input File:")
    inFileLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)

    inFileTxt = Tkinter.Entry(stepOne)
    inFileTxt.grid(row=0, column=1, columnspan=7, sticky="WE", pady=3)

    inFileBtn = Tkinter.Button(stepOne, text="Browse ...")
    inFileBtn.grid(row=0, column=8, sticky='W', padx=5, pady=2)

    inFileLbl = Tkinter.Label(stepOne, text="Select the Second Input File:")
    inFileLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)

    inFileTxt = Tkinter.Entry(stepOne)
    inFileTxt.grid(row=1, column=1, columnspan=7, sticky="WE", pady=3)

    inFileBtn = Tkinter.Button(stepOne, text="Browse ...")
    inFileBtn.grid(row=1, column=8, sticky='W', padx=5, pady=2)


    outFileLbl = Tkinter.Label(stepOne, text="Save Output File to:")
    outFileLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)

    outFileTxt = Tkinter.Entry(stepOne)
    outFileTxt.grid(row=2, column=1, columnspan=7, sticky="WE", pady=2)

    outFileBtn = Tkinter.Button(stepOne, text="Browse ...")
    outFileBtn.grid(row=2, column=8, sticky='W', padx=5, pady=2)

    form.mainloop()



import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
os.chdir("C:\\Users\\Ganesh\\RandPython")
from sklearn.datasets import make_classification, make_blobs

from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
# Load the cancer data
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
                                                   random_state = 0)
# Call the Logisitic Regression function
clf = LogisticRegression().fit(X_train, y_train)
fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
# Fit a model
clf = LogisticRegression().fit(X_train, y_train)

# Compute and print the Accuray scores
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
y_predicted=clf.predict(X_test)
# Compute and print confusion matrix
confusion = confusion_matrix(y_test, y_predicted)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, y_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, y_predicted)))

## Accuracy of Logistic regression classifier on training set: 0.96
## Accuracy of Logistic regression classifier on test set: 0.96
## Accuracy: 0.96
## Precision: 0.99
## Recall: 0.94
## F1: 0.97
