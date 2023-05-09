#!/usr/bin/env python
# coding: utf-8

# # Import necessary packages

import argparse

import pandas as pd
import datetime
import os

from hpc_runtime_prediction.model_optimization import optimize_testing_window


# Get split time and utput file name from arguments

parser = argparse.ArgumentParser()
parser.add_argument('--split-time', type=str, help='Split time in formate "YYYY-MM-DD"')
parser.add_argument('--output-file', type=str, help='Output file name')
args = parser.parse_args()

split_time = [pd.Timestamp(args.split_time)]
output_file = args.output_file

# # Import Eagle data

filepath = os.path.join('../data/', 'eagle_data.pkl')
eagle_df = pd.read_pickle(filepath)


# # Optimize Testing Window
# * Use the follwing features:
#     * Wallclock Req
#     * Nodes Req
#     * Processsors Req
#     * GPUs Req
# * Use optimal number of training days (100). 
# * Check testing days from 1 to 31 in increments of 1.
# * Check split times from 2020-01-01 to 2022-12-31 in increments of 1 week.

optimal_training_window = 100

r2_test, rmse_test = optimize_testing_window(eagle_df, split_time, optimal_training_window)

optimize_testing_window_df = pd.DataFrame({'testing_window': list(r2_test.keys()), 'r2': list(r2_test.values()), 'rmse': list(rmse_test.values())})

filename = output_file + '.pkl'
filepath = os.path.join('../results/testing_window/', filename)
optimize_testing_window_df.to_pickle(filepath)