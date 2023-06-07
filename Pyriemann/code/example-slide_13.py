
import numpy as np
from moabb.paradigms import P300
from moabb.datasets import bi2012
from pyriemann.estimation import XdawnCovariances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from pyriemann.classification import TSclassifier

# choose the dataset to get the data
dataset = bi2012(Training=True, Online=False)
# instantiate the paradigm that goes with the dataset
paradigm = P300()
# get the epochs
X, y, meta = paradigm.get_data(dataset, subjects=[1])

# instantiate the pipeline
pipeline = make_pipeline(
    XdawnCovariances(nfilter=4, estimator='lwf'),
    TSclassifier(metric='riemann', clf=LDA()))
# run 5-fold cross-validation
kf = KFold(n_splits=5)
scr = []
for train_idx, test_idx in kf.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]    
    pipeline.fit(X_train, y_train)
    scr.append(pipeline.score(X_test, y_test))
print(np.mean(scr))