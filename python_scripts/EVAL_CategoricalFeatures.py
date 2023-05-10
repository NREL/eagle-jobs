#!/usr/bin/env python
# coding: utf-8

# # Import necessary packages

import argparse

import pandas as pd
import datetime
import os

from eagle_jobs.model_optimization import optimize_categorical_features


# Get split time and utput file name from arguments

parser = argparse.ArgumentParser()
parser.add_argument('--split-time', type=str, help='Split time in format "YYYY-MM-DD"')
parser.add_argument('--output-file', type=str, help='Output file name')
args = parser.parse_args()

split_time = [pd.Timestamp(args.split_time)]
output_file = args.output_file

# # Import Eagle data

filepath = os.path.join('../data/', 'eagle_data.parquet')
eagle_df = pd.read_parquet(filepath)

# ### Optimize Categorical Features + Encoding
# * With optimal train days, test days, and numerical feature set, test the following encodings:
#     * Label encoding: Label all categorical features
#     * Target encoding: Get target encoding for all categorical features
#     * Onehot (top n) Encoding: One-hot encode top n instances of categorical features
#         * Top 20 users
#         * Top 6 partitions
#         * Top 15 accounts
#         * Top 20 name
#         * Top 20 work_dir
# * Run all combinations of User, Account, Partition, QOS, Name, and Work Dir (2^6 = 64 combinations)
# * Check split times from 2020-01-01 to 2022-12-31 in increments of 1 week.

# In[7]:

optimal_training_window = 100
optimal_testing_window = 1
optimal_numerical_features = ('wallclock_req', 'nodes_req', 'processors_req', 'gpus_req', 'mem_req')

optimal_features = dict()
for encoding in ['label','onehot','target']:
    r2_cat_feat, rmse_cat_feat = optimize_categorical_features(eagle_df, split_time, optimal_training_window, optimal_testing_window, optimal_numerical_features, encoding)

    optimize_categorical_features_df = pd.DataFrame({'features': list(r2_cat_feat.keys()), 'r2': list(r2_cat_feat.values()), 'rmse': list(rmse_cat_feat.values())})

    filename = output_file + '_' + encoding + '.pkl'
    filepath = os.path.join('../results/categorical_features/', filename)
    optimize_categorical_features_df.to_pickle(filepath)