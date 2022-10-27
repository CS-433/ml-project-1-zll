import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from helpers import load_csv_data
from helpers import create_csv_submission


# load dataset
y_train, tx_train, ids_train = load_csv_data("train.csv")
y_test, tx_test, ids_test = load_csv_data("test.csv")
           

# data preprocessing 
# extract usable data by replacing -999 in feature matrix with mean of this feature
for col in range(tx_train.shape[1]):
    null_index = np.where(tx_train[:,col] == -999.)[0]
    data_clean = [x for x in tx_train[:,col] if x != -999.]  
    col_mean = np.mean(data_clean)
    tx_train[null_index, col] = col_mean

    
# delete feature columns with correlation larger than 0.95
corr = np.corrcoef(x_train.T)
mask = np.zeros_like(corr, dtype = bool)
mask[np.triu_indices_from(mask)] = True
delete_index = list(set(np.where(corr >= 0.95)[0]))
x_train = delete_related_features(x_train, delete_index)
x_train = fix_null_value(x_train)

    
# data preprocessing -- mean subtraction and normalization   
tx_train = (tx_train - np.mean(tx_train,axis=0)) / np.std(tx_train,axis=0)


# data preprocessing -- build polynomial extension for input features
degree = 3
tx_train = build_poly(tx_train, degree)


# add offset to the feature matrix tx_train and tx_test
tx_train = np.c_[np.ones((tx_train.shape[0], 1)), tx_train]
tx_test = np.c_[np.ones((tx_test.shape[0], 1)), tx_test]


# mapping labels from {-1, 1} to {0, 1} in order to apply logistic regression
y_train[np.where(y_train == -1)] = 0
y_train = y_train.reshape(len(y_train),1)
num = y_train.shape[0]


# set the hyper-parameters
initial_w = np.array([[0] for i in range(np.shape(tx_train)[1])])
lambda_ = 1
max_iters = 10000
gamma = 1e-10


# choose the best method to obtain predicted labels of our test set
w, loss = mean_squared_error_gd(y_train, tx_train, initial_w, max_iters, gamma)
y_test = predict_labels(w, tx_test)


# map the predicted labels back to {-1, 1}
y_test[np.where(y_test == 0)] = -1


# transform predicted labels into csv file for submission
create_csv_submission(ids_test, y_test, "submission_binary_classification.csv")