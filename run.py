import numpy as np
import matplotlib.pyplot as plt

from implementations import *
from helpers import load_csv_data
from helpers import create_csv_submission


def main():

    # load dataset
    y_train, tx_train, ids_train = load_csv_data("train.csv")
    _, tx_test, ids_test = load_csv_data("test.csv")


    # data preprocessing 
    # extract usable data by replacing -999 in feature matrix with mean of this feature
    tx_train = fix_null_value(tx_train)
    tx_test = fix_null_value(tx_test)


    # delete feature columns with correlation larger than 0.8
    corr = np.corrcoef(tx_train.T)
    mask = np.zeros_like(corr, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = 0
    delete_index = list(set(np.where(np.abs(corr) >= 0.75)[0]))
    tx_train = delete_related_features(tx_train, delete_index)
    tx_test = delete_related_features(tx_test, delete_index)


    # data preprocessing -- mean subtraction and normalization   
    tx_train = (tx_train - np.mean(tx_train,axis=0)) / np.std(tx_train,axis=0)


    # data preprocessing -- build polynomial extension for input features
    degree = 3
    tx_train = build_poly(tx_train, degree)
    tx_test = build_poly(tx_test, degree)


    # add offset to the feature matrix tx_train and tx_test
    tx_train = np.c_[np.ones((tx_train.shape[0], 1)), tx_train]
    tx_test = np.c_[np.ones((tx_test.shape[0], 1)), tx_test]


    # mapping labels from {-1, 1} to {0, 1} in order to apply logistic regression
    y_train[np.where(y_train == -1)] = 0


    # set the hyper-parameters
    initial_w = np.array([[0] for i in range(np.shape(tx_train)[1])])
    lambda_ = 1e-5


    # choose the best method to obtain predicted labels of our test set
    w, loss = ridge_regression(y_train, tx_train, lambda_)
    y_test = predict_labels(w, tx_test)


    # map the predicted labels back to {-1, 1}
    y_test[np.where(y_test == 0)] = -1


    # transform predicted labels into csv file for submission
    create_csv_submission(ids_test, y_test, "submission_binary_classification_corr0.75.csv")

    

if __name__ == "__main__":
    main()