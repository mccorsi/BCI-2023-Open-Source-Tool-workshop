import warnings

# load numerical libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# load moabb functions
import moabb
from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery

# load pyriemann stuff
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# choose the dataset to get the data
dataset = BNCI2014001()
# instantiate the paradigm that goes with the dataset
paradigm = LeftRightImagery()
# get the epochs
X, y, meta = paradigm.get_data(dataset, subjects=[1])
# estimate the covariances
covs = Covariances(estimator='lwf').fit_transform(X)
# instantiate the classifier
clf = MDM(metric='riemann')
# run 5-fold cross-validation
kf = KFold(n_splits=5)
scr = []
for train_idx, test_idx in kf.split(covs):
    covs_train, y_train = covs[train_idx], y[train_idx]
    covs_test, y_test = covs[test_idx], y[test_idx]    
    clf.fit(covs_train, y_train)
    scr.append(clf.score(covs_test, y_test))
print(np.mean(scr))