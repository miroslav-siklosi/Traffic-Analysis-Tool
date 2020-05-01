"""
MACHINE LEARNING MODULES 

System Log Analysis for Anomaly Detection Using Machine Learning

MIT License

Copyright (c) 2020 Miroslav Siklosi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

''' CLASSIFICATION METHODS '''

''' Logistic Regression '''
def method_LR(data):    
    from sklearn.linear_model import LogisticRegression
    
    classifier_LR = LogisticRegression(random_state = 0)
    classifier_LR.fit(data["X_train"], data["y_train"])
    
    return classifier_LR

''' K-NN '''    
def method_KNN(data):
    from sklearn.neighbors import KNeighborsClassifier
    
    classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier_KNN.fit(data["X_train"], data["y_train"])
    
    return classifier_KNN

''' SVM ''' #This one can take too much time to process
def method_SVM(data):
    from sklearn.svm import SVC
    
    classifier_SVM = SVC(kernel = 'linear', random_state = 0)
    classifier_SVM.fit(data["X_train"], data["y_train"])
    
    return classifier_SVM

''' Kernel SVM '''
def method_kSVM(data):
    from sklearn.svm import SVC
    
    classifier_kSVM = SVC(kernel = 'rbf', random_state = 0)
    classifier_kSVM.fit(data["X_train"], data["y_train"])
    
    return classifier_kSVM

''' Naive Bayes '''
def method_NB(data):
    from sklearn.naive_bayes import GaussianNB
    
    classifier_NB = GaussianNB()
    classifier_NB.fit(data["X_train"], data["y_train"])
    
    return classifier_NB

''' Decision Tree Classification '''
def method_DTC(data):
    from sklearn.tree import DecisionTreeClassifier
    
    classifier_DTC = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier_DTC.fit(data["X_train"], data["y_train"])
    
    return classifier_DTC

''' Random Forest Classification '''
def method_RFC(data):
    from sklearn.ensemble import RandomForestClassifier
    
    classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier_RFC.fit(data["X_train"], data["y_train"])
    
    return classifier_RFC


''' CLUSTERING METHODS '''

''' K-Means Machine Learning Method '''
def method_KMeans(data):
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters = data["n_labels_y"], init = 'k-means++', random_state = 42)
    
    # Predicting the Test set results
    y_kmeans = kmeans.fit_predict(data["X"])
    
    return y_kmeans

''' Hierarchical Clustering '''
def method_HC(data):
    from sklearn.cluster import AgglomerativeClustering
    
    hc = AgglomerativeClustering(n_clusters = data["n_labels_y"], affinity = 'euclidean', linkage = 'ward')
    
    # Predicting the Test set results
    y_hc = hc.fit_predict(data["X"])
    
    return y_hc

''' DEEP LEARNING METHODS '''
''' ARTIFICAL NEURAL NETWORK METHOD '''
def method_ANN(data):
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Initialising the ANN
    classifier_ANN = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier_ANN.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu', input_dim = 79))
    
    # Adding the hidden layers
    h_layers = 1
    for i in range(h_layers):
        classifier_ANN.add(Dense(output_dim = 39, init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier_ANN.add(Dense(output_dim = data["n_labels_y_train"], init = 'uniform', activation = 'softmax'))
    
    # Compiling the ANN
    classifier_ANN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier_ANN.fit(data["X_train"], data["encoded_y_train"], batch_size = 10, epochs = 10)
    
    return classifier_ANN