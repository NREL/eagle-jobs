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
<img width="935" alt="image" src="https://media.github.nrel.gov/user/2146/files/a165045a-61a0-41de-a7e1-8eb071173686">

Per-Partition Results for Classification Models:
<img width="928" alt="image" src="https://media.github.nrel.gov/user/2146/files/9533e915-dddb-467a-8e64-e79e10f22eaf">
