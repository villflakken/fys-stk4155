import os
import sys
import pretty_errors  # available with pip via `pip install pretty_errors`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
sys.path.insert(2, '.')  # To keep my linter silenced

"""
*** Mathematical tools
"""


def compute_FrankeFunction(x, y):
    """ Computes the Franke function """
    term1 = 0.75*np.exp(-0.25*(9*x - 2)**2 - 0.25*(9*y - 2)**2)
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-0.25*(9*x - 7)**2 - 0.25*(9*y - 3)**2)
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4


def compute_MSE(y, ytilde):
    """ Mean Squared Error evaluation """
    n = float(y.size)  # Assuming 1 dimension
    return np.sum((y - ytilde)**2)/n


def compute_R2(y, ytilde):
    """ Computes R**2 score """
    ymean = np.mean(y)
    sum1 = np.sum((y - ytilde)**2)
    sum2 = np.sum((y - ymean)**2)
    return 1 - sum1/sum2


def compute_variance(vec):
    """ Computes the variance of a 1-dimensional vector """
    return sum(vec-vec.mean()**2)/float(vec.size)


def compute_n_predictors_2dim(n_poly):
    """
    Number of elements in beta (predictors);
    '2dim' refers to there being x and y as data dimensions.
    """
    # l = int((n_poly+1)*(n_poly+2)/2)    # Morten's code example
    l = np.sum(np.arange(n_poly+2))     # A quick rephrasing
    # #*   Curious for a generalized method, I attempted to rephrase Morten's example
    # #* to apply with more than 2 dimensions of data (our case: x & y)
    # l_gen = np.prod([(n_poly + i)/i for i in range(1, n_data_dimensions+1)])
    # # ^ but this failed both for n_data_dimensions = 1 and 3,
    # # _except_ in the specific case of (n_poly = 2, n_data_dimensions = 3).
    # #   Further attempts to expand upon his formulation were fruitless.
    # #   I must however now waste less time pondering this, and instead continue;
    # # using the remaining simplest formulation.
    return l


def create_X_2dim(x, y, n_poly):
    """ Create X as a design matrix of x and y"""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    #* Create the matrix
    n_len = len(x)
    l = compute_n_predictors_2dim(n_poly)
    X = np.ones((n_len, l))
    for i in range(1, n_poly+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)
    return X


def my_little_scaler(input_matrix, input_array):
    """
    Inspired from the documentation of SKL, I wrote this instead,
    to avoid potential problems related to singular matrices.
    """
    output_matrix = input_matrix.copy()
    output_array = input_array.copy()
    # The omitted column should only contain the number 1.
    output_matrix[:, 1:] -= output_matrix[:, 1:].mean(axis=0)
    output_array -= output_array.mean()
    return output_matrix, output_array


def my_train_test_splitter(X_mat, y_arr, test_size=0.2, seed=None):
    """
    My own formulation on how to train, test, and split the data.
        X_mat       :   n-times-p Design matrix, 
                            with row-equations to shuffle and split.
        y_arr       :   n-dim array with output of the hard data.
        test_size   :   The desired ratio of test-data vs. train-data.
    """
    n_rows = y_arr.size
    indexes = np.arange(n_rows)
    # Create row indeces and shuffle them
    if seed != None:
        np.random.seed(seed)
        pass
    np.random.shuffle(indexes)
    # Extract indexes for each type
    test_index_begins = int(test_size*float(y_arr.size))
    train_indexes = indexes[test_index_begins:]
    test_indexes = indexes[:test_index_begins]
    # ^ Doesn't matter whether the first part or the last part of the indexes are used;
    #   it's all shuffled, anyway.
    X_mat_train = X_mat[train_indexes].copy()  # New arrays are actually elaborately
    y_arr_train = y_arr[train_indexes].copy()  # memory-referencing values in old array,
    X_mat_test = X_mat[test_indexes].copy()  # unless copied like here.
    y_arr_test = y_arr[test_indexes].copy()
    return X_mat_train, X_mat_test, y_arr_train, y_arr_test, train_indexes, test_indexes


def compute_train_test_indexes(n_rows, test_size=0.2, seed=None):
    """
    Simplified version of `my_train_test_splitter`.
    """
    indexes = np.arange(n_rows)
    # Create row indeces and shuffle them
    if seed != None:
        np.random.seed(seed)
        pass
    np.random.shuffle(indexes)
    # Extract indexes for each type
    test_index_begins = int(test_size*float(n_rows))
    train_indexes = indexes[test_index_begins:]
    test_indexes = indexes[:test_index_begins]
    return train_indexes, test_indexes


def compute_beta_OLS(X_mat, y_arr):
    """ Matrix operations to find beta and ytilde """
    return np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T @ y_arr


def compute_variance_sample(vec_data, vec_model, n_len, p_len):
    """
    Computes sample variance w.r.t.:
        n_len       : number of observations
        p_len       : array length of predictors (beta)
        vec_data    : the raw data
        vec_model   : the regression approximation
    """
    return (1/(n_len - p_len - 1))*np.sum((vec_data - vec_model)**2)


def confidence_interval_sample(X_mat, beta, y_data, ytilde, n_len, p_len,
                               bool_print_loop=False):
    """ Computes a confidence interval from the estimates of the estimate """
    # First get covariance matrix:
    covance_mat = compute_variance_sample(
        y_data, ytilde, n_len, p_len)*np.linalg.pinv(X_mat.T @ X_mat)
    # The diagonal of the covariance matrix are the variances of beta: \Var(beta_i)**2
    variance_beta = np.diag(covance_mat)
    # Making the confidence interval for a 2*sigma (~95.4%) interval
    ci_beta = np.zeros((p_len, 2))
    for i in range(p_len):
        ci_beta[i] = (beta[i] - 2*np.sqrt(variance_beta[i]/n_len),
                      beta[i] + 2*np.sqrt(variance_beta[i]/n_len))
    # Confidence interval done!
    # A printing section to verify values
    if bool_print_loop:
        for i in range(p_len):
            print("i                                 :", i,
                  "beta[i]                           :", beta[i],
                  "variance_beta[i]                  :", variance_beta[i],
                  "2*np.sqrt(variance_beta[i]/n_len) :", 2*np.sqrt(variance_beta[i]/n_len),
                  "beta[i] +/- conf_int              :", ci_beta[i][::-1])
            print()
        pass
    return ci_beta


def my_Ridge_regression(X_mat, y_arr, lambda_):
    """
    Executes a Ridge regression, 
        introducing the infinitessimal term lamb as a so-called
        ridge hyperparameter, in order to construct beta, 
        and by extension the array of predicted values.
    """
    XTX = X_mat.T @ X_mat
    n_XTX = len(XTX)
    XTX_Ridge = np.linalg.pinv(XTX + lambda_*np.identity(n_XTX))
    beta_Ridge = XTX_Ridge @ X_mat.T @ y_arr
    y_pred_Ridge = X_mat @ beta_Ridge
    return XTX_Ridge, beta_Ridge, y_pred_Ridge

if __name__ == "__main__":
    pass