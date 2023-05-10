#!/usr/bin/env python
# coding: utf-8

# # Import necessary packages

import argparse

import pandas as pd
import datetime
import os

from eagle_jobs.model_optimization import optimize_numerical_features


# Get split time and utput file name from arguments

parser = argparse.ArgumentParser()
parser.add_argument('--split-time', type=str, help='Split time in formate "YYYY-MM-DD"')
parser.add_argument('--output-file', type=str, help='Output file name')
args = parser.parse_args()

split_time = [pd.Timestamp(args.split_time)]
output_file = args.output_file

# # Import Eagle data

filepath = os.path.join('../data/', 'eagle_data.parquet')
eagle_df = pd.read_parquet(filepath)

# # Optimize Feature Set
# * Possible Features:
#     * Wallclock Req
#     * Nodes Req
#     * Processsors Req
#     * GPUs Req
#     * Req Mem
#     * User (categorical)
#     * Account (categorical)
#     * Partition (categorical)
#     * QOS (categorical)
#     * Work Dir (categorical)
#     * Name (categorical)

# ### Optimize Numerical Features
# * Assume we should use Wallclock Req as a minimum feature set.
# * Use optimal number of training & testing days
# * In addition to wallclock, run all combinations of Nodes, Processors, GPUs, and Mem (2^4 = 16 combinations)

optimal_training_window = 100
optimal_testing_window = 1

r2_num_feat, rmse_num_feat = optimize_numerical_features(eagle_df, split_time, optimal_training_window, optimal_testing_window)

optimize_numerical_features_df = pd.DataFrame({'features': list(r2_num_feat.keys()), 'r2': list(r2_num_feat.values()), 'rmse': list(rmse_num_feat.values())})

filename = output_file + '.pkl'
filepath = os.path.join('../results/numerical_features/', filename)
optimize_numerical_features_df.to_pickle(filepath)