#!/usr/bin/env python
# coding: utf-8

# # Import necessary packages

import argparse

import pandas as pd
import datetime
import os

from hpc_runtime_prediction.model_optimization import optimize_training_window


# Get split time and utput file name from arguments

parser = argparse.ArgumentParser()
parser.add_argument('--split-time', type=str, help='Split time in format "YYYY-MM-DD"')
parser.add_argument('--output-file', type=str, help='Output file name')
args = parser.parse_args()

split_time = [pd.Timestamp(args.split_time)]
output_file = args.output_file

# # Import Eagle data

filepath = os.path.join('../data/', 'eagle_data.pkl')
eagle_df = pd.read_pickle(filepath)

# # Get split times for model method optimization

# # Optimize Training Window
# * Use the follwing features:
#     * Wallclock Req
#     * Nodes Req
#     * Processsors Req
#     * GPUs Req
# * Testing window = 30. 
# * Checks training window of size 1 day, then 5 to 180 days in increments of 5 (i.e. 1,5,10,15...,175,180).

r2_train, rmse_train = optimize_training_window(eagle_df, split_time, model_type='TFIDF')

optimize_training_window_df = pd.DataFrame({'training_window': list(r2_train.keys()), 'r2': list(r2_train.values()), 'rmse': list(rmse_train.values())})

filename = output_file + '.pkl'
filepath = os.path.join('../results/training_window/raw/tfidf/', filename)
optimize_training_window_df.to_pickle(filepath)