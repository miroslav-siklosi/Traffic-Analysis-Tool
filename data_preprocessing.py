"""
DATA PREPROCESSING
System Log Analysis for Anomaly Detection Using Machine Learning
MIT License
Copyright (c) 2020 Miroslav Siklosi
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Method for importing unlabelled dataset
def import_unlabelled_dataset(filename):
    # Load the dataset
    dataset = pd.read_csv(filename)
    
    # Load dataset into matrix of independant variables 
    X_test = dataset.iloc[:, list(range(4, 6)) + list(range(7, 84))].values

    # Taking care of missing and incorrect data
    SUM = 0
    MAX = 0
    COUNT = 0
    
    # Count average values in columns 15 and 16
    for i, row in enumerate(X_test):
        for j in [15, 16]:
            sx = str(float(X_test[i,j])).lower()
            if  (sx != "nan" and sx != "inf"):
                SUM = SUM + X_test[i,j]
                if X_test[i,j] > MAX:
                    MAX = X_test[i,j]
                COUNT = COUNT + 1
    
    AVERAGE = SUM/COUNT
    
    for i, row in enumerate(X_test):
        for j in [15, 16]:
            sx = str(float(X_test[i,j])).lower()
            if  sx == "nan":
                X_test[i, j] = AVERAGE    
            if  sx == "inf":
                X_test[i, j] = MAX
    
    return {"dataset": dataset, "X_test": X_test}

# Method for importing labelled dataset
def import_dataset(filename, split):    
    # Load the dataset
    dataset = pd.read_csv(filename)
    
    # Splitting the dataset into independent and dependent variables
    X = dataset.iloc[:, list(range(4, 6)) + list(range(7, 84))].values
    y = np.array([0 if val == "BENIGN" else 1 for val in dataset.iloc[:, -1].values])
    
    # Taking care of missing and incorrect data
    SUM = 0
    MAX = 0
    COUNT = 0
    
    # Count average values in columns 15 and 16
    for i, row in enumerate(X):
        for j in [15, 16]:
            sx = str(float(X[i,j])).lower()
            if  (sx != "nan" and sx != "inf"):
                SUM = SUM + X[i,j]
                if X[i,j] > MAX:
                    MAX = X[i,j]
                COUNT = COUNT + 1
    
    AVERAGE = SUM/COUNT
    
    for i, row in enumerate(X):
        for j in [15, 16]:
            sx = str(float(X[i,j])).lower()
            if  sx == "nan":
                X[i, j] = AVERAGE    
            if  sx == "inf":
                X[i, j] = MAX
    
    # Splitting the dataset into the Training set and Test set   
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y

    return {"dataset": dataset, 
            "X": X, "y": y, 
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
            }