from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb

import os
import pandas as pd
from config import *
from dataloading import *
from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost_params import params

def train_xgboost(data_train, data_val, predictor_columns, target_column):
    predictors_train = np.array(data_train[predictor_columns])
    predictors_val = np.array(data_val[predictor_columns])

    dtrain = xgb.DMatrix(predictors_train, label=np.array(data_train[target_column]))
    dval = xgb.DMatrix(predictors_val, label=np.array(data_val[target_column]))

    # Train with evaluation set and early stopping
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=evallist, early_stopping_rounds=10, verbose_eval = False)

    return bst

def test_xgboost(data_test, predictor_columns, target_column, bst):
    predictors_test = np.array(data_test[predictor_columns])
    dtest = xgb.DMatrix(predictors_test, label=np.array(data_test[target_column]))
    preds = bst.predict(dtest)
    return preds

def run_xgboost_with_results(seed, data, predictor_columns, target_column):
    data_train, data_val, data_test = create_train_val_test_split(data, seed, test_size)
    bst = train_xgboost(data_train, data_val, predictor_columns, target_column)
    preds = test_xgboost(data_test, predictor_columns, target_column, bst)
    rmse = compute_rmse(preds, data_test[target_column])
    return rmse

def run_xg_boost_with_results_for_all_target_columns(seed, data):
    results_list = []
    result = run_xgboost_with_results(seed, data, predictor_columns, 'H_AVERAGE')
    results_list.append(result)
    result = run_xgboost_with_results(seed, data, predictor_columns, 'D_AVERAGE')
    results_list.append(result)
    result = run_xgboost_with_results(seed, data, predictor_columns, 'VOLUME')
    results_list.append(result)
    return results_list




    