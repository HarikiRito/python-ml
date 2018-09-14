import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
def flip(m):
    if type(m) is np.ndarray:
        r = flip_ndarray(m)
    else:
        r = flip_ndarray(np.asarray(m))
    return r


def flip_ndarray(m):
    return np.rot90(m)[::-1]

def categorical(X, i:int):
    """
    Categorize columns
    :param X: (The Matrix itself)
    :param i: Int (Index of Matrix)
    :return: X
    """
    X[:, i] = LabelEncoder().fit_transform(X[:, i])
    return OneHotEncoder(categorical_features=[i]).fit_transform(X).toarray()
