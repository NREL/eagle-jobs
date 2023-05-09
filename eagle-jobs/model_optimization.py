import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import datetime
import os

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from category_encoders.glmm import GLMMEncoder

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from hpc_runtime_prediction.operation_support import train_test_split
from hpc_runtime_prediction.operation_support import normalize_columns

from hpc_runtime_prediction.data_preprocessing import label_encode_columns
from hpc_runtime_prediction.data_preprocessing import onehot_with_other


import warnings
warnings.filterwarnings('ignore')

def optimize_training_window(df, split_times, model_type='XGBoost'):
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 30
    training_windows = [1] + list(range(5,181,5))
    for training_window in training_windows:
        r2_dict[training_window] = list()
        rmse_dict[training_window] = list()
    
    for split_time in split_times:
        for training_window in training_windows:
            train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
            if len(train_df) < 2 or len(test_df) < 2:
                continue
            train_features = train_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            test_features = test_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            if model_type == 'XGBoost':
                model = xgb.XGBRegressor()
                model.fit(train_features, train_target)
            elif model_type == 'NN':
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=[train_features.shape[1]]),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(1)
                ])
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
                history = model.fit(train_features, train_target, batch_size=10000, epochs=10,\
                                validation_data=(test_features, test_target), callbacks=[early_stopping], verbose=0)
            elif model_type == 'TFIDF':
                features = ['wallclock_req_XSD_duration','nodes_req','processors_req','gpus_req','mem_req','user','account','partition','qos','work_dir','name']
                X_train = ''

                for feature in features:
                    X_train += train_df[feature].astype('str')
                y_train = train_df.run_time.values
                vect = TfidfVectorizer(max_features=600)
                X_train_vec = vect.fit_transform(X_train.values)
                
                model = LinearRegression(n_jobs=-1)
                model.fit(X_train_vec, y_train)
                
                X_test = ''
                for feature in features:
                    X_test += test_df[feature].astype('str')
                X_test = X_test.values
                test_features = vect.transform(X_test)
                test_target = test_df.run_time.values
                
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[training_window].append(r2)
            rmse_dict[training_window].append(rmse)
            
            print(f'Split time: {split_time}, training window: {training_window}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_testing_window(df, split_times, training_window):
    r2_dict = dict()
    rmse_dict = dict()
    testing_windows = range(1,61)
    for testing_window in testing_windows:
        r2_dict[testing_window] = list()
        rmse_dict[testing_window] = list()
    
    for split_time in split_times:
        train_df = df[df.end_time.between(split_time - datetime.timedelta(training_window),\
                                          split_time, inclusive='left')]
        if len(train_df) < 2:
            continue
        train_features = train_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
        train_target = train_df['run_time']
        
        model = xgb.XGBRegressor()
        model.fit(train_features, train_target)
            
        for testing_window in testing_windows:
            test_df = df[df.submit_time.between(split_time + datetime.timedelta(testing_window - 1),\
                                                split_time + datetime.timedelta(testing_window))]
            if len(test_df) < 2:
                continue
            test_features = test_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            test_target = test_df['run_time']
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[testing_window].append(r2)
            rmse_dict[testing_window].append(rmse)
            
            print(f'Split time: {split_time}, testing window: {testing_window}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_numerical_features(df, split_times, training_window, testing_window):
    features = ['nodes_req', 'processors_req', 'gpus_req', 'mem_req']
    feature_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(features, r) for r in range(0, len(features) + 1)))
    feature_combinations = [('wallclock_req',) + f for f in feature_combinations]
    
    r2_dict = dict()
    rmse_dict = dict()
    for f in feature_combinations:
        r2_dict[f] = list()
        rmse_dict[f] = list()
    
    for split_time in split_times:
        train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
        if len(train_df) < 2 or len(test_df) < 2:
            continue
        
        for f in feature_combinations:
            train_features = train_df[list(f)]
            test_features = test_df[list(f)]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            model = xgb.XGBRegressor()
            model.fit(train_features, train_target)
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[f].append(r2)
            rmse_dict[f].append(rmse)
            
            print(f'Split time: {split_time}, features: {f}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_categorical_features(df, split_times, training_window, testing_window, numerical_features, encoding):
    features = ['user','account','partition','qos','work_dir','name']
    
    if encoding == 'label':
        label_encode_columns(df, features)
    elif encoding == 'onehot':
        n_values = [20,15,6,3,20,20]
        encoded_column_names = onehot_with_other(df, features, n_values)
    
    feature_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(features, r) for r in range(0, len(features) + 1)))
    feature_combinations = [numerical_features + f for f in feature_combinations]
    
    r2_dict = dict()
    rmse_dict = dict()
    for fc in feature_combinations:
        r2_dict[fc] = list()
        rmse_dict[fc] = list()
    
    for split_time in split_times:
        train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
        if len(train_df) < 2 or len(test_df) < 2:
            continue
                
        if encoding == 'target':
            encoder = GLMMEncoder(cols=features, random_state=42)
            train_encoded = encoder.fit_transform(train_df[list(numerical_features)+features], train_df.run_time)
            test_encoded = encoder.transform(test_df[list(numerical_features)+features])
            train_df = pd.concat([train_encoded, train_df.run_time], axis=1)
            test_df = pd.concat([test_encoded, test_df.run_time], axis=1)
            
        for fc in feature_combinations:
            if encoding == 'onehot':
                categorical_features = list(set(fc) & set(features))
                categorical_columns = list()
                for cf in categorical_features:
                    categorical_columns += encoded_column_names[cf]
                numerical_features = list(set(fc) - set(features))
                final_features = numerical_features + categorical_columns
            else:
                final_features = fc
            
            train_features = train_df[list(final_features)]
            test_features = test_df[list(final_features)]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            model = xgb.XGBRegressor()
            model.fit(train_features, train_target)
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[fc].append(r2)
            rmse_dict[fc].append(rmse)
            
            print(f'Encoding: {encoding}, Split time: {split_time}, features: {fc}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict

def average_runtime_algorithm(train_df, user_df, n):
    if user_df is None or len(user_df) == 0:
        return 600
    avg_runtime = user_df.run_time.tail(n).mean()
    return avg_runtime

def optimize_recent_jobs(df, split_time, n_max):
    split_time = pd.Timestamp(split_time)
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 1
    training_window = 100

    train_df, test_df = train_test_split(df, split_time, training_window, testing_window)

    user_dataframes = dict()
    for user in train_df.user.unique():
        user_dataframes[user] = train_df[train_df.user == user]
    for user in test_df.user.unique():
        if user not in user_dataframes:
            user_dataframes[user] = None
            
    for n in range(1,n_max+1):
        y_test = test_df.run_time.values
        y_pred = test_df.apply(lambda x: average_runtime_algorithm(train_df, user_dataframes[x['user']], n), axis=1).values
       
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2_dict[n] = r2
        rmse_dict[n] = rmse
        print(f'Split time: {split_time}, n: {n}, r2: {r2}, rmse: {rmse}')

    return r2_dict, rmse_dict



def similar_jobs_algorithm(train_df, user_jobs, job_row, n):
    if user_jobs is None:
        return train_df.run_time.mean()
    if len(user_jobs[0]) < n + 2:
        n = len(user_jobs[0]) - 2
        if n < 1:
            return train_df.run_time.mean()
    similarity = np.linalg.norm(user_jobs[0] - job_row, axis=-1)
    similar_job_indices = np.argpartition(similarity, n+1)[:n+1]
    ind = user_jobs[1]
    return train_df.loc[ind[similar_job_indices]].run_time.mean()

def optimize_similar_jobs(df, split_time, n_max):
    df['submit_time_seconds'] = (df['submit_time'] - np.datetime64('1970-01-01T00:00:00')) // np.timedelta64(1, 's')
    df['wallclock_req_normalized'] = df['wallclock_req']
    df['submit_time_normalized'] = df['submit_time_seconds']
    features = ['nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'wallclock_req_normalized', 'submit_time_normalized']
    df, scaler = normalize_columns(df, features)
    
    split_time = pd.Timestamp(split_time)
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 1
    training_window = 100

    train_df, test_df = train_test_split(df, split_time, training_window, testing_window)

    user_dataframes = dict()
    for user in train_df.user.unique():
        user_dataframes[user] = list()
        user_dataframes[user].append(train_df[train_df.user == user][features])
        user_dataframes[user].append(user_dataframes[user][0].index)
    for user in test_df.user.unique():
        if user not in user_dataframes:
            user_dataframes[user] = None
            
    for n in range(1,n_max+1):
        y_test = test_df.run_time.values
        y_pred = test_df.apply(lambda x: similar_jobs_algorithm(train_df, user_dataframes[x['user']], x[features].to_numpy().astype(float), n), axis=1)
        
        if len(y_test) < 2 or len(y_pred) < 2:
            continue
            
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2_dict[n] = r2
        rmse_dict[n] = rmse
        print(f'Split time: {split_time}, n: {n}, r2: {r2}, rmse: {rmse}')

    return r2_dict, rmse_dict