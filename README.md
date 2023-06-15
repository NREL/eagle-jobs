# Eagle Jobs
#### *An HPC dataset of 11M+ jobs and a Codebase for HPC Runtime Prediction Models*
This repository is the home of a sample dataset of jobs run on the Eagle HPC system at the National Renewable Energy Laboratory in Golden, Colorado, USA, as well as a limited alpha version of the codebase used to analyze Eagle data and develop HPC runtime prediction models. The full 11M+ job dataset can be downloaded at [https://data.openei.org/submissions/5860](https://data.openei.org/submissions/5860). The data is available in parquet (253.1 MB) and compressed CSV (115.6 MB) formats. To use this dataset with the Jupyter notebooks in this repository, clone this repo and put the parquet file in the `data` directory.

Below are the key details of this repository:

* The sample dataset is the first 1000 jobs submitted to Eagle on or after January 1, 2019. 

* All potentially sensitive information (such as user names, project titles, and account information) has been anonymized to generic labels (e.g. user003, account098, etc.).

* The data is saved in the `data` directory in csv, json, and pkl formats.

* The codebase consists of the `hpc_runtime_prediction` package as well as Jupyter notebooks and Python scripts for data visualization, building, training, testing, and optimizing a runtime prediction model.

* More info about Eagle: [https://www.nrel.gov/hpc/eagle-system.html](https://www.nrel.gov/hpc/eagle-system.html)

## Getting Started
### 1 - *Clone this repository*
### 2 -  *Create the conda environment from the `requirements.txt` file*
From the repo root directory, run:
```
conda create --name hpc-rt --file requirements.txt
```
### 3 - *Activate the conda environment*
```
conda activate hpc-rt
```
### 4 - *Build the `hpc_runtime_prediction` package*
From the repo root directory, run:
```
python setup.py install
```
### 5 - *Open Jupyter Lab*
```
jupyter lab
```
This will open Jupyter Lab in your default browser.
### 6 - *Using Jupyter Lab*
- A detailed Jupyter Lab tutorial can be found [here](https://jupyterlab.readthedocs.io/en/stable/).
- Open the `notebooks` folder
- The notebooks in this directory are:
  - `Eagle_Data_Visualization`: Use this notebook to analyze the Eagle dataset.
  - `Runtime_Variation`: Use this notebook to analyse the variation in HPC job runtime.
  - `Combine_Results`: Use this notebook to combine results generated with the Python scripts (see details on these scripts below).
  - `Model_Optimization`: Use this notebook to optimize a runtime prediction model's feature set, training, and testing window.
  - `Hyperparameter_Optimization`: Use this notebook to optimize the model hyperparameters with [Optuna](https://optuna.readthedocs.io/en/stable/).
  - `HPO_Visualization`: Use this notebook to visualize the results of Hyperparameter Optimization.
  - `Final_Model_Analysis`: Use this notebook to analyze the results of a runtime prediction model.
- *Note:* None of these notebooks are set up for the sample data set, and they will not work well with the sample data set due to its limited size.
  
### 7 - *Parallelizing the Workload with Pytho Scripts*
The Python scripts in the python_scripts directory are created to allow the model optimization process to be parallelized. This is necessary for large datasets, such as the Eagle dataset, because the iterative optimization process used in this codebase can result in the training aand testing of tens of thousands of models. This process can be parallelized by training and testing models in batches based on a specified split time (see the [Wiki documentation](https://github.com/NREL/eagle-jobs/wiki/Model-Optimization) for more details on this process).

### 8 - *Deactivate the conda environment*
To deactivate the environment, run:
```
conda deactivate
```

# Credit

Kevin and Dmitry, as employees of Alliance for Sustainable Energy, LLC, the manager and operator of the National Renewable Energy Laboratory.

# License

Refer to the file called: LICENSE.
