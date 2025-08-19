SETUP:
created environment named 'timeshap' with python=3.10
with the following command:
conda create -n timeshap python=3.10
if you are not using Anaconda, pip should also work.

With Anaconda, run the following in your environment:
conda install -c sebp scikit-survival (tested with 0.22.2)
conda install shap (0.47.2) (used to be 0.42.1)
conda install matplotlib (3.10.0)
conda install ipython (8.30.0)

the above packages normally leads to the
installation of the following base packages as well:
- bottleneck-1.3.7
- numpy-1.26.3
- numpy-base-1.26.3
- scikit-learn-1.3.0
- pillow 11.3.0