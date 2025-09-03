# timeSHAP

A Python project (soon to be, package) for _time-dependent_ [SHAP](https://shap.readthedocs.io/en/latest/) (SHapley Additive exPlanations) explanations in survival models. This repository provides scripts and utilities for generating synthetic survival data, running time-dependent SHAP explanations for [Random Survival Forest](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html) models, and visualizing the generated explanations.

## Features
- Generate synthetic survival data for benchmarking
- Compute and visualize time-dependent SHAP values
- Support for scikit-survival models
- Example scripts and figures for reproducibility

## Setup

### 1. Environment Creation
Create a new conda environment named `timeshap` with e.g. Python 3.10:

```sh
conda create -n timeshap python=3.10
conda activate timeshap
```

If you are not using Anaconda, you can use `pip` to install the required packages (see below).

### 2. Install Required Packages
With Anaconda, run the following commands in your environment:

```sh
conda install -c sebp scikit-survival>=0.22
conda install shap>=0.47
conda install matplotlib>=3.10
conda install ipython>=8.30
```

These packages will also install the following dependencies:
- bottleneck
- numpy
- scikit-learn
- pillow

If using pip, install the equivalent versions:

```sh
pip install "scikit-survival>=0.22" "shap==0.47" "matplotlib==3.10.0" "ipython==8.30.0"
```

## Repository Structure

- `main_script_run.py` — Main script to run timeSHAP analysis
- `generate_surv_data.py` — Script to generate synthetic survival data (being tested)
- `environment.yml` — Conda environment specification (alternative to manual setup)
- `FLChain-single-event-imputed/` — Example dataset (csv files)
- `saved_data_*.pkl` — Example saved datasets
- `draft-figures/`, `figures/` — Output figures and plots when draft setting is on (with smaller data)
- `other-material-and-slides/` — Additional figures and slides

## Usage

1. **Activate the environment:**
   ```sh
   conda activate timeshap
   ```
2. **Run the main script:**
   ```sh
   python main_script_run.py
   ```
   This will perform the timeSHAP analysis and generate output figures in the `figures/` directory.

3. **Generate synthetic data (optional):**
   ```sh
   python generate_surv_data.py
   ```

## Output
- Figures and plots are saved in the `figures/` and `draft-figures/` directories.
- Example datasets are provided in `FLChain-single-event-imputed/`.

## Notes
- The code is tested with the package versions listed above. Other package and Python versions should work but are not guaranteed.
- For questions or issues, please open an issue on GitHub.
