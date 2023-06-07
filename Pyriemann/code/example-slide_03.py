import warnings

# load numerical libraries
import numpy as np
import pandas as pd

# load moabb functions
import moabb
from moabb.datasets import Weibo2014
from moabb.paradigms import LeftRightImagery

# load pyriemann tools
from pyriemann.estimation import Covariances

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# choose dataset
dataset = Weibo2014()
# choose paradigm
paradigm = LeftRightImagery()
# extract epochs from the dataset following the paradigm setup
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
# estimate covariance matrices from the epochs
covs = Covariances(estimator='lwf').fit_transform(X)