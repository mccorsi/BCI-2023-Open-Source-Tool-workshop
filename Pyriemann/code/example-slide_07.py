import warnings

# load numerical libraries
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

# load moabb functions
import moabb
from moabb.datasets import Weibo2014, Zhou2016
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery

# load pyriemann stuff
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, TSclassifier

# load graphical utilities
import matplotlib
import seaborn
seaborn.reset_orig()
import matplotlib.pyplot as plt

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# choose which datasets to consider from MOABB
datasets = [Weibo2014(), Zhou2016()]
paradigm = LeftRightImagery()

# instantiate the pipelines
pipelines = {}
pipelines['cov-mdm-euc'] = make_pipeline(
	Covariances(estimator='lwf'), MDM(metric='euclid'))
pipelines['cov-mdm-rie'] = make_pipeline(
	Covariances(estimator='lwf'), MDM(metric='riemann'))
pipelines['cov-tsp-lda'] = make_pipeline(
	Covariances(estimator='lwf'), TSclassifier(metric='riemann'))

# setup the evaluation procedure
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=False)

# run the valuation process
results = evaluation.process(pipelines)

# generate figure with the results average across subjects
fig, ax = plt.subplots(facecolor='white', figsize=(6.4, 4.8))
plt.subplots_adjust(left=0.20, right=0.95)
npipelines = len(pipelines.keys())
cmap = plt.cm.summer
norm = matplotlib.colors.Normalize(vmin=0, vmax=npipelines-1)
yticks = []
for d, dataset in enumerate(evaluation.datasets):
	dataset_code = dataset.code
	print(dataset)
	scores = results[results['dataset'] == dataset_code]
	for i, pipelinei in enumerate(list(pipelines.keys())):
		scorei = scores[scores['pipeline'] == pipelinei].mean().score
		yi = i + (npipelines+2)*d
		ax.barh(yi, scorei, height=1.00, color=cmap(norm(i)), align='center')		
		yticks.append(yi)
ndatasets = len(evaluation.datasets)
yticks = np.array(yticks)		
ax.set_xlim(0.4, 1.05)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(['0.4', '', '0.6', '', '0.8', '', '1.0'])
ax.set_ylim(-1.5,(npipelines+2)*ndatasets-1.5)
ylim = ax.get_ylim()
for ticki in [0.6, 0.8]:
	ax.plot([ticki, ticki], ylim, ls='--', color=[0, 0, 0, 0.40], lw=1.0)
ax.set_yticks(yticks)
ax.set_yticklabels(
	list(pipelines.keys()) + list(pipelines.keys()), fontsize=11)
for d in range(ndatasets-1):
	ax.plot([0.4, 1.05], [9.5 + (npipelines+2)*d, 9.5 + (npipelines+2)*d],
	c='k', lw=0.8)
ax.plot([1.0, 1.0], ylim, c='k', lw=0.8)	
for di, dataset in enumerate(evaluation.datasets):
	ax.text(
		1.025, 1 + di*(npipelines+2), dataset.code, rotation=-90, ha='center',
	    va='center', fontsize=11)
ax.set_title('Comparing three simple pipelines', fontsize=14)
ax.set_xlabel('AUC', fontsize=12)
fig.savefig(fname='example-slide_07.pdf', format='pdf')