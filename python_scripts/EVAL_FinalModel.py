#!/usr/bin/env python
# coding: utf-8

# In[3]:


import datetime
import os
import pandas as pd

from xgboost import XGBRegressor

from eagle_jobs.operation_support import train_test_split
from eagle_jobs.data_preprocessing import label_encode_columns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


def evaluate_model(df, split_times):
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 1
    training_window = 100
    
    pred_vs_act_df = pd.DataFrame(columns=['runtime_pred', 'runtime_act'])
    
    for split_time in split_times:
        train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        if len(train_df) < 2 or len(test_df) < 2:
            continue
            
        features = ['wallclock_req','processors_req','mem_req','nodes_req','gpus_req','user','partition']
        train_features = train_df[features]
        test_features = test_df[features]
        train_target = train_df['run_time']
        test_target = test_df['run_time']
        
        params = { # From Optuna HPO
            'n_estimators': 168,
            'max_depth': 7,
            'learning_rate': 0.3968571956999504,
            'gamma': 0.640232768439118,
            'subsample': 0.747747407403972,
            'colsample_bytree': 0.6280085182287491
        }

        model = XGBRegressor(**params)
        model.fit(train_features, train_target)

        y_pred = model.predict(test_features)
        
        new_df = pd.DataFrame({'runtime_pred': y_pred, 'runtime_act': test_target})
        
        pred_vs_act_df = pd.concat([pred_vs_act_df, new_df], ignore_index=True)
        
        r2 = r2_score(test_target, y_pred)
        rmse = mean_squared_error(test_target, y_pred, squared=False)

        r2_dict[split_time] = r2
        rmse_dict[split_time] = rmse
        
        rmse_list = list(rmse_dict.values())
        r2_list = list(r2_dict.values())

        print(f'Split time: {split_time}, r2: {r2:.3f}, rmse: {rmse:.0f}, Avg R2: {sum(r2_list)/len(r2_list):.3f}, Avg RMSE: {sum(rmse_list)/len(rmse_list):.0f}')
       
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    filename = 'pred_vs_act_df_'+now+'.pkl'
    pred_vs_act_df.to_pickle(filename)
    
    return r2_dict, rmse_dict


if __name__ == '__main__':
    filepath = os.path.join('../data/', 'eagle_data.parquet')
    eagle_df = pd.read_parquet(filepath)

    categorical_features = ['user','partition']
    label_encode_columns(eagle_df, categorical_features)

    filtered_df = eagle_df[['wallclock_req','processors_req','mem_req','nodes_req','gpus_req',\
                            'user','partition','run_time','submit_time','end_time']]

    start_time = pd.Timestamp('2019-02-01')
    end_time = pd.Timestamp('2023-02-01')
    split_times = pd.date_range(start_time, end_time, freq='D')

    r2, rmse = evaluate_model(filtered_df, split_times)

    eval_final_model_df = pd.DataFrame({'training_window': list(r2.keys()), \
                                        'r2': list(r2.values()), 'rmse': list(rmse.values())})

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    filename = 'eval_final_model_df_'+now+'.pkl'
    eval_final_model_df.to_pickle(filename)

