
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
import os
import shutil
import csv
import scipy.stats as stats

from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import mpmath as mpm
import math


class Kernel:
    """ 
    Base class for kernel functions.
    """
    def __init__(self, kernel_name: str):
        """
        Initialize the kernel function.

        Parameters:
        - kernel_name: (str) Name of the kernel function.
        """
        self.kernel_name = kernel_name
        Ker = {'Normal' : self.Normal_pdf, 'Epanechnikov': self.Epanechnikov_pdf, 'Tophat': self.Tophat_pdf, 'Exponential': self.Exponential_pdf, 'Linear': self.Linear_pdf, 'Cosine': self.Cosine_pdf}

        if kernel_name in Ker:
            self.pdf = Ker[kernel_name]
        else:
            raise ValueError("Invalid kernel name. Supported kernels: {}".format(Ker.keys()))

    def Normal_pdf(self, x):
        """ Normal kernel function"""
        return 1/np.sqrt(2*np.pi)*np.exp(-np.power(x, 2)/2)
    
    def Epanechnikov_pdf(self, x):
        """ Epanechnikov kernel function"""
        return 3/4*np.maximum(0, 1-np.power(x, 2))

    def  Tophat_pdf(self, x):
        """ Tophat kernel function"""
        return 0.5*(np.abs(x) <= 1)

    def Exponential_pdf(self, x):
        """ Exponential kernel function"""
        return 0.5*np.exp(-np.abs(x))

    def Linear_pdf(self, x):
        """ Linear kernel function"""
        return (1-np.abs(x))*(np.abs(x) <= 1)

    def Cosine_pdf(self, x):
        """ Cosine kernel function"""
        return np.pi/4*np.cos(np.pi*x/2)*(np.abs(x) <= 1)


def compute_knn_bandwidth(DATA: pd.DataFrame, feature_idx: int, k: int, h_0: float):
    """
    Compute adaptive bandwidths h_i = h_0 * d_k(X_i) where d_k(X_i) is the 
    distance to the k-th nearest neighbor for a given feature.

    Parameters:
    - DATA: (pd.DataFrame) Input data containing covariates.
    - feature_idx: (int) Index of the feature column.
    - k: (int) Number of nearest neighbors to consider.
    - h_0: (float) Base bandwidth scaling factor.

    Returns:
    - h_x: (np.ndarray) Adaptive bandwidths for each data point.
    """
    X_feature = DATA.iloc[:, feature_idx + 2].values.reshape(-1, 1)  # Extract feature column
    tree = cKDTree(X_feature)  # Build k-d tree for fast nearest neighbor lookup
    distances, _ = tree.query(X_feature, k=k + 1)  # k+1 because first neighbor is itself
    d_k = distances[:, -1]  # Get distance to k-th nearest neighbor
    return h_0 * d_k

def CFP_NW(Y: np.ndarray, DATA: pd.DataFrame, kernels: list, hs: list, adaptive_h=False) -> float:
    """Compute the conditional Failure Probability (CFP) using Nadaraya-Watson estimator. 
        The CFP is the bernouilli parameter p(Y=1|X) = E[Y|X] with
        Y the variable giving the disruption (1) of a generator or not (0)
        X the vector of the covariates, here the temperature and the demand
        We use the Nadaraya-Watson estimator to compute this regression
        INPUTS:
        - Y : (array) the variables at which we compute p 
        - DATA : (dataframe) the historical data with columns:
            - "number_disruptions" : number of disrupted generators on a given day, given area (e.g texas)
            - "number_generators_available" : number of available generators in the area at the begining of the day
            - Y1 : the first covariate (e.g temperature)
            - Y2 : the second covariate (e.g demand)
            - Y... : other covariates if needed
        - kernels : (list of functions) the kernel functions pdf used to compute the regression
        - hs : (list of floats) the (fixed) bandwidths used for each kernel. If adaptive_h is True, it is the base bandwidth
        - adaptive_h : (False, int, array) if int, we compute the adaptive kernel bandwidth with the k nearest neighbor distance (BREIMAN, L., MEISEL, W. and PURCELL, E. (1977)), if array it is already the distance
        OUTPUT:
        - p : (float in [0,1]) the estimated probability of disruption
    """
    ny = DATA.shape[1]-2 # the number of covariates
    if len(kernels) != ny:
        raise ValueError("The number of kernels must be equal to the number of covariates (current: {})".format(len(kernels)))
    if len(hs) != ny:
        raise ValueError("The number of bandwidths must be equal to the number of covariates (current: {})".format(len(hs)))
    if len(Y) != ny:
        raise ValueError("The number of inputs must be equal to the number of covariates (current: {})".format(len(Y)))
    if DATA.columns[0] != 'number_disruptions':
        raise ValueError("The first column of DATA must be the number of disruptions on the day. Please name the column 'number_disruptions' to confirm.")
    if DATA.columns[1] != 'number_generators_available':
        raise ValueError("The second column of DATA must be the number of available generators on the day. Please name the column 'number_generators_available' to confirm.")
    # Compute adaptive bandwidth if needed
    if type(adaptive_h)==bool:
        if adaptive_h==False:
            h_x = np.array(hs)
        else:
            raise SystemExit("'adaptive_h' must be an integer or an array of distances or False boolean")
    elif type(adaptive_h)==int:
        h_x = np.array([compute_knn_bandwidth(DATA, i, adaptive_h, hs[i]) for i in range(ny)])
    else:
        h_x = adaptive_h

    K = [kernels[i]((DATA.iloc[:, i + 2] - Y[i]) / h_x[i]) for i in range(ny)]
    K = np.prod(K, axis=0)
    
    p = np.sum(K*DATA["number_disruptions"])/np.sum(K*DATA["number_generators_available"])
    return p


def compute_knn_bandwidth(DATA: pd.DataFrame, feature_idx: int, k: int, h_0: float):
    """
    Compute adaptive bandwidths h_i = h_0 * d_k(X_i) where d_k(X_i) is the 
    distance to the k-th nearest neighbor for a given feature.

    Parameters:
    - DATA: (pd.DataFrame) Input data containing covariates.
    - feature_idx: (int) Index of the feature column.
    - k: (int) Number of nearest neighbors to consider.
    - h_0: (float) Base bandwidth scaling factor.

    Returns:
    - h_x: (np.ndarray) Adaptive bandwidths for each data point.
    """
    X_feature = DATA.iloc[:, feature_idx + 2].values.reshape(-1, 1)  # Extract feature column
    tree = cKDTree(X_feature)  # Build k-d tree for fast nearest neighbor lookup
    distances, _ = tree.query(X_feature, k=k + 1)  # k+1 because first neighbor is itself
    d_k = distances[:, -1]  # Get distance to k-th nearest neighbor
    return h_0 * d_k

