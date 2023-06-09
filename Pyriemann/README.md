# Pyriemann

[2023 International BCI meeting](https://bcisociety.org/bci-meeting/)

Workshop W5 Session 1 - Wednesday, June 7, 9:30am – 12:30pm (Sonian Forest Time)

## Overview
`pyRiemann` is a Python machine learning package based on `scikit-learn` API. 
It provides a high-level interface for processing and classification of
multivariate time series through the Riemannian geometry of symmetric positive
definite (SPD) matrices.

`pyRiemann` aims at being a generic package for multivariate time series
classification but has been designed around multichannel biosignals (like EEG,
MEG or EMG) manipulation applied to brain-computer interface (BCI),
transforming multichannel time series into covariance matrices, and classifying
them using the Riemannian geometry of SPD matrices.

## Take-home messages
- Using methods from Riemannian geometry for BCI is easy with `pyriemann`
- `pyriemann` is built around `scikit-learn` and benefits from all of its tools
- We're always happy to have new **contributions** from the community!

## Resources
- The slides of my talk at the BCI meeting are available [[here](https://github.com/mccorsi/BCI-2023-Open-Source-Tool-workshop/blob/main/Pyriemann/pyRiemann-slides.pdf)]
- A video recording is available [[here](https://youtu.be/osclkBPcmlk)]
- Several pieces of **code** used for my presentation are available in the 
`/code` folder

## References
`pyriemann` website: https://pyriemann.readthedocs.io/en/latest/