# sbatch_pred: Technical Documentation

## Introduction

The `sbatch_pred` tool is designed to provide accurate predictions for HPC job runtimes and queue times. It uses Machine Learning (ML) models to deliver these predictions with associated uncertainty estimates. This document presents a detailed technical analysis of the models and methodologies used in `sbatch_pred`.

## Model Overview

`sbach_pred` employs two types of models:

1. **Classification Models**: These models categorize jobs into different wait time classes. Historical confusion matrices are maintained to understand the distribution of actual wait times for each predicted class.

2. **Regression Models**: These models predict continuous wait times. The tool introduces random noise into job features and distributions of actual wait times from similar jobs to quantify uncertainty.

## Results Analysis

Our analysis shows that:

- The accuracy of queue time predictions improves significantly when the model is given knowledge of the system state (e.g. number of jobs waiting in a queue, number of nodes currently in use, etc.),  with similar improvements for classification and regression models.
- Using perfect knowledge of job runtime improves performance when compared to models trained with the user estimate of job runtime.
- Using the job runtime predicted with ML decreased the error in queue time predicted by the regression models.

# Detailed Results
Per-Partition Results for Regression Models:
<img width="916" alt="image" src="https://github.com/NREL/eagle-jobs/assets/77375297/bfc93e36-7b44-4b4a-bca1-5fe8726e90b2">

Per-Partition Results for Classification Models:
<img width="914" alt="image" src="https://github.com/NREL/eagle-jobs/assets/77375297/57af98e9-3ad2-4509-88e7-45d63cd4d9eb">

Confusion Matrices for System-State Feature Set Variations
![image](https://github.com/NREL/eagle-jobs/assets/77375297/82b860b6-331d-4dde-8527-07d7b3027623)


