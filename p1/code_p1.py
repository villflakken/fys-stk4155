import os
import sys
import pretty_errors # available with pip via `pip install pretty_errors`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import math_functions as mf
import plot_functions as pf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

#!!! Execution flow of each part are collected at the bottom of this program

"""
*** Misc shorthand
"""

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


def print_comparison_designX_details(X_mat, y_arr):
    """ 
    Methods and measurements used to compare my own implementations against.
    """
    print("\nDetails pertaining to the Design Matrix & Linear Regression (for comparison)")
    # Split in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y_arr, test_size=0.2)
    clf = skl.LinearRegression().fit(X_train, y_train)

    # The mean squared error and R2 score
    print("MSE before scaling: {:.2f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score before scaling {:.2f}".format(clf.score(X_test, y_test)))
    print() # Visual aid

    # Properties pertaining to the scaled version
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature min values before scaling:\n {}".format(X_train.min(axis=0)))
    print("Feature max values before scaling:\n {}".format(X_train.max(axis=0)))
    print("Feature min values after scaling:\n {}".format(X_train_scaled.min(axis=0)))
    print("Feature max values after scaling:\n {}".format(X_train_scaled.max(axis=0)))
    clf = skl.LinearRegression().fit(X_train_scaled, y_train)
    print("MSE after  scaling: {:.2f}".format(mean_squared_error(clf.predict(X_test_scaled), y_test)))
    print("R2 score for  scaled data: {:.2f}".format(clf.score(X_test_scaled, y_test)))
    print() # Visual aid
    pass


"""
*** Script flow
"""

def part_a(seed):
    """
        Part a) of the assignment, or, alternatively:
    How to train your drago--- I mean, regression model

        Constitutes script to execute Part a)'s necessary operations.
        Variable name declaration might have been excessively explicit;
    actually caused me headaches later.
    """
    #* Declaring component variables and data
    n_poly = 5      # Polynomial size
    n_len  = 1000   # Data point length/no. of observations
    p_len = mf.compute_n_predictors_2dim(n_poly)
    # n_data_dimensions = 2   # used in an earlier, more ambitious version
    # Base meshgrid data:
    x = np.sort(np.random.uniform(0, 1, n_len))
    y = np.sort(np.random.uniform(0, 1, n_len))
    x, y = np.meshgrid(x, y)
    # z's data w/noise:
    sigma_noise, mu_noise = 0.25, 0
    noise_z = sigma_noise*np.random.randn(n_len, n_len) + mu_noise
    z = mf.compute_FrankeFunction(x, y) + noise_z
    
    #* Need those flattened 1d arrays later, though!:
    x_data_1d = np.ravel(x)
    y_data_1d = np.ravel(y)
    z_data_1d = np.ravel(z)

    #* Make a plot to visualize z
    # pf.plot_FrankeFunction(x, y, z)
    # plt.show()

    #* Create design matrix and beta
    my_X = mf.create_X_2dim(x_data_1d, y_data_1d, n_poly)
    my_X_OLS = my_X.copy()
    print(f"(a): X.shape = {my_X_OLS.shape}")
    # ^ Declare a copy of X as a precaution, so the original is not 
    #     perturbed/modified unintended w.r.t. view/scope in functions.
    beta_OLS = mf.compute_beta_OLS(X_mat=my_X_OLS, y_arr=z_data_1d)
    ztilde_OLS = my_X_OLS @ beta_OLS # predicted model's values
    print(f"(a): beta.shape = {beta_OLS.shape}")

    #* Confidence intervals
    mf.confidence_interval_sample(my_X_OLS, beta_OLS, z_data_1d, ztilde_OLS,
                                n_len, p_len)#, bool_print_loop=True)

    #* Again, but w/scaling & train/test-splitting the design matrix
    my_X_OLS_scaled, z_scaled = mf.my_little_scaler(my_X, z_data_1d)

    my_X_OLS_scaled_train, my_X_OLS_scaled_test, \
        z_scaled_train, z_scaled_test,           \
        train_indexes, test_indexes =            \
            mf.my_train_test_splitter(my_X_OLS_scaled, z_scaled, test_size=0.2, seed=seed)
    # Some obvious constructs to make
    beta_OLS_scaled_train = mf.compute_beta_OLS(my_X_OLS_scaled_train, z_scaled_train)
    beta_OLS_scaled_test  = mf.compute_beta_OLS(my_X_OLS_scaled_test , z_scaled_test )
    ztilde_OLS_scaled_train = my_X_OLS_scaled_train @ beta_OLS_scaled_train
    ztilde_OLS_scaled_test  = my_X_OLS_scaled_test  @ beta_OLS_scaled_test
    # The tiny test
    ztilde_OLS_scaled_Xtest_btrain = my_X_OLS_scaled_train @ beta_OLS_scaled_test
    ztilde_OLS_scaled_Xtrain_btest = my_X_OLS_scaled_test @ beta_OLS_scaled_train 
    # ^ the last one's the one we're mostly interested in, I think

    if True == 1:
        print(f"(a):--- --- ---\n",
            f"n_len   = {n_len}",
            f"p_len   = {p_len}")
        # print(" my_X:")
        # print("my_X_OLS_scaled_train.shape:", my_X_OLS_scaled_train.shape)
        # print("my_X_OLS_scaled_test.shape :", my_X_OLS_scaled_test.shape)
        # print(" beta:")
        # print("beta_OLS_scaled_train.shape:", beta_OLS_scaled_train.shape)
        # print("beta_OLS_scaled_test.shape :", beta_OLS_scaled_test.shape)
        # print(" ztilde:")
        # print("ztilde_OLS_scaled_train.shape:", ztilde_OLS_scaled_train.shape)
        # print("ztilde_OLS_scaled_test.shape :", ztilde_OLS_scaled_test.shape)
        # print(" ztilde_complicated:")
        # print("ztilde_OLS_scaled_Xtest_btrain.shape:", ztilde_OLS_scaled_Xtest_btrain.shape)
        # print("ztilde_OLS_scaled_Xtrain_btest.shape:", ztilde_OLS_scaled_Xtrain_btest.shape)
        print(
            f"\n *** OLS: MSE & R2 measurements ***",
            f"\n  ( whole sample       /            straight ):",
            f"\n* MSE : {mf.compute_MSE(z_data_1d                , ztilde_OLS)}",
            f"\n* R2  : {mf.compute_R2( z_data_1d                , ztilde_OLS)}",
            f"\n  ( training subsample /            straight ):",
            f"\n* MSE : {mf.compute_MSE(z_data_1d[train_indexes] , ztilde_OLS_scaled_train)}",
            f"\n* R2  : {mf.compute_R2( z_data_1d[train_indexes] , ztilde_OLS_scaled_train)}",
            f"\n  ( testing subsample  /            straight ):",
            f"\n* MSE : {mf.compute_MSE(z_data_1d[test_indexes ] , ztilde_OLS_scaled_test)}",
            f"\n* R2  : {mf.compute_R2( z_data_1d[test_indexes ] , ztilde_OLS_scaled_test)}\n",
            f"\n  ( training subsample / X_test @ beta_train ):",
            f"\n* MSE : {mf.compute_MSE(z_data_1d[train_indexes] , ztilde_OLS_scaled_Xtest_btrain)}",
            f"\n* R2  : {mf.compute_R2( z_data_1d[train_indexes] , ztilde_OLS_scaled_Xtest_btrain)}",
            f"\n  ( testing subsample  / X_train @ beta_test ):",
            f"\n* MSE : {mf.compute_MSE(z_data_1d[test_indexes ] , ztilde_OLS_scaled_Xtrain_btest)}",
            f"\n* R2  : {mf.compute_R2( z_data_1d[test_indexes ] , ztilde_OLS_scaled_Xtrain_btest)}\n")
        pass
    return


def part_b(seed):
    """
    Bias-variance, and resampling
    """
    #! Make a figure similar to figure 2.11 of Hastie et al
    #* Declaring component variables and data
    n_polys = np.arange(32)  # Polynomial sizes for the 2.11-replication plot
    n_len = 1000             # Data point length/no. of observations
    p_len = np.array([mf.compute_n_predictors_2dim(n_p) for n_p in n_polys])
    print(f"(b): p_len = {p_len}")
    # Base meshgrid data:
    x = np.sort(np.random.uniform(0, 1, n_len))
    y = np.sort(np.random.uniform(0, 1, n_len))
    x, y = np.meshgrid(x, y)
    # z's data w/noise:
    sigma_noise, mu_noise = 0.25, 0
    noise_z = sigma_noise*np.random.randn(n_len, n_len) + mu_noise
    z = mf.compute_FrankeFunction(x, y) + noise_z

    #* Need those flattened 1d arrays later, though!:
    x_data_1d = np.ravel(x)
    y_data_1d = np.ravel(y)
    z_data_1d = np.ravel(z)
    
    # Lists to store our results
    X_dict      = {'train': {}, 'test': {}}
    beta_dict   = {'train': {}, 'test': {}}
    ztilde_dict = {'train': {}, 'test': {}}
    mse_trains  = np.zeros(len(n_polys))
    mse_tests   = np.zeros(len(n_polys))

    #* Create design matrices, betas, and predictions with varying degrees of polynomials
    for i in np.arange(len(n_polys)):
        #- Create design matrix and split it
        print(f"(b): In loop i = {i+1:>2} / {len(n_polys):<2} | {'~'+str(int(i/len(n_polys)*100)):>3}%")
        X_to_split = mf.create_X_2dim(x_data_1d, y_data_1d, n_polys[i])
        train_inds, test_inds = mf.compute_train_test_indexes(n_rows=n_len, test_size=0.2, seed=seed)
        X_train                 = X_to_split[train_inds]
        X_test                  = X_to_split[test_inds ]
        X_dict["train"][i]      = X_train
        X_dict["test" ][i]      = X_test
        # print(f"(b): X_train.shape = {str(X_train.shape):>10} --- (n_p={n_polys[i]:>2})")
        # print(f"(b): X_test.shape  = {str(X_test.shape ):>10} --- (n_p={n_polys[i]:>2})")
        #- Compute beta
        beta_train              = mf.compute_beta_OLS(X_train, z_data_1d[train_inds])
        beta_test               = mf.compute_beta_OLS(X_test,  z_data_1d[test_inds] )
        beta_dict["train"][i]   = beta_train
        beta_dict["test" ][i]   = beta_test
        #- Compute the model's predicted output
        ztilde_train            = X_train @ beta_train
        ztilde_test             = X_test  @ beta_test
        ztilde_dict["train"][i] = ztilde_train
        ztilde_dict["test" ][i] = ztilde_test
        #- Compute MSE
        mse_trains[i] = mf.compute_MSE(z_data_1d[train_inds], ztilde_train)
        mse_tests [i] = mf.compute_MSE(z_data_1d[test_inds ], ztilde_test )
        continue

    # Now we can plot the MSEs against one another
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(n_polys, mse_trains, label="Training sample")
    ax.plot(n_polys, mse_tests , label="Testing sample" )
    ax.legend(loc='best')
    # plt.show()
    save_fig("b - train-test MSE vs n_poly (similar to 2dot11 of Hastie)")

    pass
 
def part_c(seed):
    """
    Script to execute this part.
    """

    lambda_ = 0.1
    #   - Ridge
    # my_XTX_Ridge, beta_Ridge, ztilde_Ridge = my_Ridge_regression(X_ols, z_data_1d, lambda_)
    X_r = X.copy()
    XTX = X_r.T @ X_r
    n_XTX = len(XTX)
    XTX_Ridge = np.linalg.pinv(XTX + lambda_*np.identity(n_XTX))
    beta_Ridge = XTX_Ridge @ X_r.T @ z_data_1d
    ztilde_Ridge = X_r @ beta_Ridge

    MSE_Ridge = mf.compute_MSE(z_data_1d, ztilde_Ridge)

    # Operations complete!
    pass
 
def part_d(seed):
    """
    Script to execute this part.
    """

    # Operations complete!
    pass
 
def part_e(seed):
    """
    Script to execute this part.
    """

    # Operations complete!
    pass
 
def part_f(seed):
    """
    Script to execute this part.
    """

    # Operations complete!
    pass
 
def part_g(seed):
    """
    Script to execute this part.
    """

    # Operations complete!
    pass

 
def main(seed):
    """ Primary flow of this script """
    # Execution
    part_a(seed)
    part_b(seed)
    # part_c(seed)
    # part_d(seed)
    # part_e(seed)
    # part_f(seed)
    # part_g(seed)

if __name__ == "__main__":
    # Where to save the figures and data files
    FIGURE_ID        = "Results/FigureFiles"
    PROJECT_ROOT_DIR = "Results"
    DATA_ID          = "DataFiles/"
    seed = 4155
    np.random.seed(seed)

    if not os.path.exists(PROJECT_ROOT_DIR):
        os.mkdir(PROJECT_ROOT_DIR)

    if not os.path.exists(FIGURE_ID):
        os.mkdir(FIGURE_ID)

    # if not os.path.exists(DATA_ID):
    #     os.mkdir(DATA_ID)
    
    main(seed)
    pass
