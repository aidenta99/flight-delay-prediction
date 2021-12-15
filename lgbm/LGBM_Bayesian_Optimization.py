#!/usr/bin/env python
# coding: utf-8

#basic tools 
import os
import numpy as np
import pandas as pd
import pickle

#tuning hyperparameters
from bayes_opt import BayesianOptimization  

#building models
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Load dataset
df_train_validation = pd.read_csv("final_train_val.csv", low_memory=False, index_col="id")
df_test = pd.read_csv("final_test.csv", low_memory=False, index_col="id")

df_train_validation = reduce_mem_usage(df_train_validation) 
df_test = reduce_mem_usage(df_test) 

# Split data for lgbm
df_train, df_validation = train_test_split(df_train_validation, test_size=0.20, random_state = 42)
X_train, y_train = df_train.drop("ARRIVAL_DELAY", axis=1), df_train["ARRIVAL_DELAY"]
X_val, y_val = df_validation.drop("ARRIVAL_DELAY", axis=1), df_validation["ARRIVAL_DELAY"]
X_test = df_test


# Define function for bayesian optimization LGBM

def bayes_parameter_opt_lgbm(X, y, init_round=15, opt_round=25, n_folds=10, random_seed=6, output_process=False):
    # prepare data
    train_data = lgbm.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgbm_eval(learning_rate, num_leaves, num_iterations, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf, min_sum_hessian_in_leaf):
        params = {'application':'regression_l2', 'metric':'mse', 'early_stopping_round': 3, 'verbosity': -1}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params['num_leaves'] = int(round(num_leaves))
        params['num_iterations'] = int(round(num_iterations))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        
        cv_result = lgbm.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=False, metrics=['l2'])
        return max(cv_result['l2-mean'])
     
    lgbmBO = BayesianOptimization(lgbm_eval, {
        'learning_rate': (0.01, 1.0),
        'num_leaves': (4, 800),
        'num_iterations': (10, 400),
        'feature_fraction': (0.1, 1.0),
        'bagging_fraction': (0.1, 1.0),
        'max_depth': (2, 10),
        'max_bin':(10,200),
        'min_data_in_leaf': (10, 400),
        'min_sum_hessian_in_leaf':(0,400),
    }, random_state=4242)
    
    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
    
    lgbmBO.maximize(init_points=init_round, n_iter=opt_round)
    
    model_mse=[]
    for model in range(len(lgbmBO.res)):
        model_mse.append(lgbmBO.res[model]['target'])
    
    # return best parameters
    return lgbmBO.res[pd.Series(model_mse).idxmax()]['target'],lgbmBO.res[pd.Series(model_mse).idxmax()]['params']

# Find and save optimal parameters
opt_params = bayes_parameter_opt_lgbm(X_train, y_train, init_round=100, opt_round=100, n_folds=10, random_seed=4224)
print(opt_params)

with open('opt_params.pickle', 'wb') as f:
    pickle.dump(opt_params, f)
