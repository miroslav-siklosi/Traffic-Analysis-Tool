"""
System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Importing the libraries
import argparse
import sys
import numpy as np
import ML_modules as ML
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import import_dataset, import_unlabelled_dataset
from keras.models import load_model

# List of ML methods flags for parser
methods_flags = (
    "LR",
    "K-NN",
    "ocSVM",
    "kSVM",
    "NB",
    "DTC",
    "RFC",
    "iF",
    "LOF",
    "K-Means",
    "HC",
    "ANN"    
)

# Create parser
parser = argparse.ArgumentParser(prog="IDS_traffic_analysis.py")
parser.add_argument("--mode", dest="mode", choices=["research", "prod"], required=True)
parser.add_argument("--command", dest="command", choices=["train", "test", "trainandtest"], required=True)
parser.add_argument("--method", dest="method", choices=methods_flags, required=True)
parser.add_argument("--source", dest="source", required=True)

#args = parser.parse_args(["--mode", "research", "--method", "iF", "--command", "trainandtest", "--source", "Datasets\sample_data.csv"])
#args = parser.parse_args(["--mode", "prod", "--method", "iF", "--command", "test", "--source", "Datasets\sample_data.csv"])
args = parser.parse_args()

# TODO remove before publishing
print(args.command)
print(args.mode)
print(args.method)
print(args.source)

# Definition of ML Methods - used in parser due to different needs of each methods
supervised = ("LR", "K-NN", "kSVM", "NB", "DTC", "RFC")
unsupervised = ("ocSVM", "iF", "LOF", "K-Means", "HC")
deepLearning = ("ANN")

# Assigning ML methods to corresponding parser flags
methods = {"LR": ML.method_LR, "K-NN": ML.method_KNN, "kSVM":  ML.method_kSVM, 
           "NB": ML.method_NB, "DTC":  ML.method_DTC, "RFC":  ML.method_RFC, 
           "ocSVM":  ML.method_ocSVM, "iF": ML.method_iF, "LOF": ML.method_LOF, 
           "K-Means":  ML.method_KMeans, "HC": ML.method_HC, "ANN":  ML.method_ANN}

# Method to print metrics in command line
def print_metrics(method, data, y_pred):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(data["y_test"], y_pred)
    print(f"Accuracy of Machine Learning method {method} is", accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(data["y_test"], y_pred)
    print(f"Precision of Machine Learning method {method} is", precision)
    # recall: tp / (tp + fn)
    recall = recall_score(data["y_test"], y_pred)
    print(f"Recall of Machine Learning method {method} is", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(data["y_test"], y_pred)
    print(f"F1-Score of Machine Learning method {method} is", f1)

# Method to print Prediction results into the text file
def print_prediction_result(data, y_pred, input_filepath):
    # [X_test, y_pred] Prediction is correct/Prediction is NOT correct
    # X_test = data["X_test"]
    y_test = data['y_test']
    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open(f"Results/prediction_result.csv", 'w') as f:
        with open(input_filepath) as input_file:
            for index, input_line in enumerate(input_file):
                if index == 0:
                    continue
                i = index - 1
                f.write(f"{input_line.rstrip()}, {y_pred[i]}, ")
                if y_test[i] == y_pred[i]:
                    f.write("Prediction is correct\n")
                else:
                    f.write("Prediction is NOT correct\n")
    print(f"Prediction results saved into prediction_result.csv")
    
# Method for saving ML weights (classifier)
def save_classifier(classifier, method):
    if method in supervised:
        output_filename = f"classifiers/classifier_{method}.joblib"
        dump(classifier, output_filename)
    elif method in deepLearning:
        output_filename = f"classifiers/classifier_{method}.h5"
        classifier.save(output_filename)
    return output_filename

# Verify if dataset to import is in correct format
def is_dataset_source(filename):
    filename = filename.lower()
    if filename.endswith(".csv"):
        return True
    elif filename.endswith(".joblib") or filename.endswith(".h5"):
        return False
    else:
        print(f"Invalid file extension on file {filename}")
        sys.exit(1)

# Method to load saved ML weight file (classifier)        
def load_classifier(filename):
    filepath = filename.lower()
    try:
        if filepath.endswith(".joblib"):
            if args.method not in supervised:
                print(f"Invalid classifier type for the {args.method} learning method")
                sys.exit(1)
            
            return load(filename)
        elif filepath.endswith(".h5"):
            if args.method not in deepLearning:
                print(f"Invalid classifier type for the {args.method} learning method")
                sys.exit(1)
            
            return load_model(filename)
        else:
            print("Classifier with unknown extension")
            sys.exit(1)
    except FileNotFoundError:
        print(f"{filepath} was not found!")
        sys.exit(1)

# PARSER (MENU)
if args.mode == "research": #RESEARCH MODE
    if args.command == "train": # TRAIN
        if args.method in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            classifier = method(data)
        else: # Classifier
            classifier = load_classifier(args.source)
        output_filename = save_classifier(classifier, args.method)
        print(f"Trained classifier saved into file {output_filename}")
    
    elif args.command == "test": # TEST
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_dataset(args.source, split=False)
        if args.method in unsupervised: # Unsupervised
            method = methods[args.method]
            y_pred = method(data)
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
        
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred, args.source) 
            
        else: # Supervised, Deep Learning
            if args.method in deepLearning:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.joblib")
                y_pred = classifier.predict(data["X_test"])
                
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred, args.source) 
            
    else: # TRAIN AND TEST
        if not is_dataset_source(args.source):
                print(f"{args.source} is not dataset with extension .csv")
                sys.exit(1)

        if args.method in unsupervised: # Unsupervised
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            y_pred = method(data)
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
        
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred, args.source)
                    
        else: # Supervised, Deep Learning
            data = import_dataset(args.source, split=True)
            method = methods[args.method]
            classifier = method(data) 
            y_pred = classifier.predict(data["X_test"])
 
            if args.method in deepLearning:
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            
            # Print results
            print(f"Confusion Matrix of Machine Learning Method {args.method}:")
            print(confusion_matrix(data["y_test"], y_pred))
            print_metrics(args.method, data, y_pred)
            print_prediction_result(data, y_pred, args.source)

else: # PRODUCTION MODE
    if args.command == "train": # TRAIN
        if args.method in unsupervised:
            print("Unsupervised does not need training...exiting")
            sys.exit(1)
        
        dataset_source = is_dataset_source(args.source)
        if dataset_source:
            data = import_dataset(args.source, split=False)
            method = methods[args.method]
            classifier = method(data)
        else: # Classifier
            classifier = load_classifier(args.source)
        output_filename = save_classifier(classifier, args.method)
        print(f"Trained classifier saved into file {output_filename}")
    
    elif args.command == "test": # TEST
        if not is_dataset_source(args.source):
            print(f"{args.source} is not dataset with extension .csv")
            sys.exit(1)
    
        data = import_unlabelled_dataset(args.source) 
        if args.method in unsupervised: # Unsupervised
            method = methods[args.method]
            y_pred = method(data)
            
            if args.method == "ocSVM" or "iF":
                for i, row in enumerate(y_pred):                    
                    if y_pred[i] == 1:
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
            
        else: # Supervised, Deep Learning
            if args.method in deepLearning:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.h5")
                y_pred = classifier.predict(data["X_test"])
                y_pred = (y_pred > 0.5)
                # Invert back to numbers
                y_pred = np.argmax(y_pred, axis = 1)
            else:
                classifier = load_classifier(f"classifiers/classifier_{args.method}.joblib")
                y_pred = classifier.predict(data["X_test"])

        labelled_dataset = np.c_[data["dataset"], ["Anomaly" if val else "Not anomaly" for val in y_pred]]
        np.set_printoptions(threshold=np.inf)
        with open(f"Results/{args.method}_labelled.csv", 'w') as f:
            for row in labelled_dataset:
                row = np.array(list(map(lambda s: s, row)))
                r = np.array2string(row, separator='\t ', max_line_width=np.inf, formatter={'str_kind': lambda x: x})
                f.write(f"{r[1:-1]}\n")
        print(f"Labelled dataset printed out to Results/{args.method}_labelled.csv")
    else: # TRAIN AND TEST
        print("Train and Test is possible only in research mode")
        sys.exit(1)