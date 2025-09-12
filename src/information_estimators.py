"""
Developped by Malo Lahogue mlahogue@mit.edu on August 2025
---
Implement methods for calcultating mutual and conditional mutual 
information between random variables based on some realizations.
Some methods are inspired by the npeet library. However it lacks 
the estimator for conditionam MI with mixed datatypes.
We implement this based on the work published by
    Zan, L.; Meynaoui, A.; Assaad, C.K.; Devijver, E.; Gaussier, E.
    A Conditional Mutual Information Estimator for Mixed Data and an Associated Conditional
    Independence Test. Entropy 2022, 24, 1234. https://doi.org/10.3390/e24091234
"""


import numpy as np
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree


############################################
########## Mutual information ##############
############################################

def mi_cc(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the mutual information I(X;Y) between two continuous random variables X and Y.
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of values for random variable X. x has dim (n, d1) where n is the number of samples and d1 is the number of features.
    - y: numpy array of values for random variable Y. y has dim (n, d2) where n is the number of samples and d2 is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Mutual information value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    return entropy_c(x, k=k, base=base) - cond_entropy_cc(x, y, k=k, base=base)

def mi_dd(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    """
    Calculate the mutual information I(X;Y) between two discrete random variables X and Y.
    
    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y.
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Mutual information value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    return entropy_d(x, base) - cond_entropy_dd(x, y, base=base)

def mi_dc(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the mutual information I(X;Y) between a discrete random variable X and a continuous random variable Y.
    Density estimation is made with the k-nearest neighbors method.

    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y. y has dim (n, d) where n is the number of samples and d is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).

    Returns:
    - Mutual information value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    x = np.asarray(x)
    Y = np.asarray(y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = Y.shape[0]

    # Small jitter to break metric ties
    Y = Y + 1e-10 * np.random.random_sample(Y.shape)

    # Classes and per-class indices
    classes, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    idx_per_class = [np.where(inv == c)[0] for c in range(len(classes))]

    # Global tree on all Y
    tree_all = build_tree(Y)

    # Per-class trees (only if class has >= 1 sample)
    trees_per_class = [build_tree(Y[idx]) if len(idx) > 0 else None for idx in idx_per_class]

    # Per-sample: class size n_xi, kth in-class radius eps_i, and global m_i within eps_i
    n_xi = counts[inv].astype(int)
    eps = np.empty(n)
    m = np.empty(n, dtype=int)

    for c, idx in enumerate(idx_per_class):
        n_c = len(idx)
        if n_c <= 1:
            # No neighbors in-class; set radius infinite and minimal m
            eps[idx] = np.inf
            m[idx] = 1
            continue

        # Query up to k+1 neighbors in-class (self at distance 0)
        kk = min(k + 1, n_c)
        dists = trees_per_class[c].query(Y[idx], k=kk)[0]
        # Take the k-th neighbor distance (exclude self). If class is tiny, last column.
        kth = dists[:, -1] if kk < (k + 1) else dists[:, k]

        # Shrink radius slightly so counts are strictly inside the ball
        radius = np.nextafter(kth, 0.0)
        eps[idx] = radius

        # Count neighbors in the FULL data within that radius; exclude self
        cnt_full = count_neighbors(tree_all, Y[idx], radius)
        m[idx] = np.maximum(cnt_full - 1, 1)

    # Ross/NPEET mixed-MI estimator:
    # I = ψ(k) - (1/n)∑ψ(n_xi) + ψ(n) - (1/n)∑ψ(m_i)
    mi = (digamma(k)
          - np.mean(digamma(n_xi))
          + digamma(n)
          - np.mean(digamma(m)))

    return float(mi / np.log(base))

def mi_cd(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the mutual information I(X;Y) between a continuous random variable X and a discrete random variable Y.
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y. y has dim (n, d) where n is the number of samples and d is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Mutual information value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    return mi_dc(y, x, k=k, base=base)

############################################
###### Conditional Mutual information ######
############################################


############################################
################ Entropy ###################
############################################

def cond_entropy_cc(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the conditional entropy H(X|Y) of a continuous random variable X given another continuous random variable Y.
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of values for random variable X. x has dim (n, d1) where n is the number of samples and d1 is the number of features.
    - y: numpy array of values for random variable Y. y has dim (n, d2) where n is the number of samples and d2 is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Conditional entropy value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    xy = np.c_[x, y]
    return entropy_c(xy, k=k, base=base) - entropy_c(y, k=k, base=base)

def cond_entropy_dd(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> float:
    """
    Calculate the conditional entropy H(X|Y) of a discrete random variable X given another discrete random variable Y.
    
    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y.
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Conditional entropy value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    xy = np.c_[x, y]
    return entropy_d(xy, base) - entropy_d(y, base)


def cond_entropy_dc(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the conditional entropy H(X|Y) of a discrete random variable X given a continuous random variable Y.
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y. y has dim (n, d) where n is the number of samples and d is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Conditional entropy value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"

    # return cond_entropy_cd(y, x, k=k, base=base) + entropy_d(x, base=base) - entropy_c(y, k=k, base=base)  # H(X|Y) = H(Y|X) + H(X) - H(Y)
    return entropy_d(x, base) - mi_dc(x, y, k=k, base=base) # numerically stronger 

    
def cond_entropy_cd(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the conditional entropy H(X|Y) of a continuous random variable X given a discrete random variable Y.
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of values for random variable X.
    - y: numpy array of values for random variable Y. y has dim (n, d) where n is the number of samples and d is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Conditional entropy value.
    """
    if len(x) == 0 or len(y) == 0: return 0.0
    assert len(x) == len(y), "Incompatible shapes"
    # Pairwise finite mask (avoid NaNs/inf breaking KNN)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    uniques, counts = np.unique(y, return_counts=True, axis=0)
    proba = counts.astype(float) / len(y)

    cond_entropy = 0.0
    for val_y, prob_y, n_y in zip(uniques, proba, counts):
        if n_y <= 1 or prob_y <= 0.0:
            # With 0 or 1 sample, the conditional differential entropy is undefined; contribute 0.
            continue
        mask = _row_equal_mask(y, val_y)
        x_y = x[mask]
        # KNN estimator requires k < n_y
        k_eff = int(min(k, max(1, n_y - 1)))

        H_x_given_y = entropy_c(x_y, k=k_eff, base=base)  # <- uses your KNN estimator
        cond_entropy += prob_y * H_x_given_y

    return cond_entropy


def entropy_d(x: np.ndarray, base: float = 2.0) -> float:
    """
    Calculate the entropy of a (set of) discrete random variable(s).
    
    Parameters:
    - x: numpy array of discrete values. x has dim (n, d) where n is the number of samples and d is the number of features.
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Entropy value.
    """
    if len(x) == 0:
        return 0.0
    uniques, counts = np.unique(x, return_counts=True, axis=0)
    proba = counts.astype(float) / len(x)
    return -np.sum(proba * np.log(proba) / np.log(base))


def entropy_c(x: np.ndarray, k:int=3, base: float = 2.0) -> float:
    """
    Calculate the entropy of a (set of) continuous random variable(s). 
    Density estimation is made with the k-nearest neighbors method.
    
    Parameters:
    - x: numpy array of continuous values. x has dim (n, d) where n is the number of samples and d is the number of features.
    - k: number of nearest neighbors to use for estimating entropy (default is 3).
    - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
    Returns:
    - Entropy value.
    """
    if len(x) == 0: return 0.0
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_elements, n_features = x.shape
    assert k <= n_elements - 1, "Set k smaller than num. samples - 1"

    x = add_noise(x) #avoid degeneracy
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


# def entropy_cd(x: np.ndarray, y: np.ndarray, k:int=3, base: float = 2.0) -> float:
#     """
#     Calculate the entropy of a continuous random variable X and a discrete random variable Y.
#     Density estimation is made with the k-nearest neighbors method.
    
#     Parameters:
#     - x: numpy array of continuous values. x has dim (n, d1) where n is the number of samples and d1 is the number of features.
#     - y: numpy array of discrete values. y has dim (n, d2) where n is the number of samples and d2 is the number of features.
#     - k: number of nearest neighbors to use for estimating entropy (default is 3).
#     - base: logarithm base for entropy calculation (default is 2 to return a quantity measures in bits).
    
#     Returns:
#     - Entropy value.
#     """
#     if len(x) == 0 or len(y) == 0: return 0.0
#     assert len(x) == len(y), "Incompatible shapes"

#     n_elements_x, n_features_x = x.shape
#     x = add_noise(x) #avoid degeneracy





############################################
################# Helpers ##################
############################################

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

# def build_tree(points):
#     if points.shape[1] >= 20:
#         return BallTree(points, metric="chebyshev")
#     return KDTree(points, metric="chebyshev")

def build_tree(points):
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.shape[1] >= 20:
        return BallTree(points, metric="minkowski", p=np.inf)
    return KDTree(points, metric="chebyshev")



def _row_equal_mask(Y: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Row-wise equality mask for 2D Y against a single row val."""
    if Y.ndim == 1:
        return Y == val  # both 1D
    # Ensure val is 1D row
    val = np.asarray(val)
    if val.ndim > 1:
        val = val.ravel()
    return np.all(Y == val, axis=1)