#!/usr/bin/env python
# coding: utf-8

#from __future__ import division, print_function # for python 2

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# helper lib
from helper_functions import *


# Load Data

data = pd.read_pickle('./data_base_train.pkl').fillna(method='ffill')
indicators = pd.read_pickle('./data_indicators_base_train.pkl').fillna(method='bfill')
tickers = pd.read_csv('./quantiacs_futures_details.csv').Ticker.tolist()

# Time-series CV for each ticker

# initialization
N_splits = 3
tss = TimeSeriesSplit(n_splits=N_splits)
weights = np.arange(1, N_splits+1) # to change based on n_splits
weights = weights / np.sum(weights) # weighted average based on size of split
results = {}
rf_param_grid = {
    'n_estimators':[80, 100, 150, 200],
    'max_features':[0.5, 0.6, 0.7, 0.8, 0.9],
    'min_samples_leaf':[2, 3, 4, 5, 6],
    'max_depth':[8, 10, 12, 15, 20]
}
rf_params = ParameterGrid(rf_param_grid)

# grid search with TS-CV, took really long
for ticker in tqdm(tickers):
    # grid search
    ticker_best_score = 1e7
    train = generate_features(data[ticker], 
                              p=4, d=1, P=2, D=0, s=5, ma=[5,40])
    train = pd.concat([train.reset_index(drop=True),
                       indicators.reset_index(drop=True)],
                      axis=1).dropna()
    for param in rf_params:
        rf = RandomForestRegressor(random_state=12345,
                                   n_jobs=2,
                                   **param)
        scores = []
        for train_ind, test_ind in tss.split(train):
            # split train test
            X = np.asarray(train.drop(columns=['TARGET']))
            y = np.asarray(train['TARGET'])
            # fit and score
            rf.fit(X[train_ind], y[train_ind])
            mse = rmse_ratio(y[test_ind], rf.predict(X[test_ind]))
            scores.append(mse)
            
        model_mse = np.dot(weights, scores)
        if model_mse < ticker_best_score:
            ticker_best_score = model_mse
            results[ticker] = {'param':param, 'mse':model_mse}

results = pd.DataFrame.from_dict(results, orient='index')
results.to_csv('rf_final_results.csv')

# Train each model with grid-searched params, rolling predict on test set

params = pd.read_csv('./rf_final_results.csv', index_col=0)

test_data = pd.read_pickle('./data_base_test.pkl').fillna(method='ffill').fillna(method='bfill')
test_indicators = pd.read_pickle('./data_indicators_base_test.pkl').fillna(method='bfill')
test_2 = pd.read_pickle('./data_stack_test.pkl').fillna(method='ffill').fillna(method='bfill')
test_indicators_2 = pd.read_pickle('./data_indicators_stack_test.pkl').fillna(method='bfill')

test_data = pd.concat([test_data, test_2])
test_indicators = pd.concat([test_indicators, test_indicators_2])

# train and predict on test set with given params per ticker
models = {}
for ticker in tqdm(tickers):
    # model
    rf_params = eval(params.loc[ticker, 'param'])
    # ensure integer for critical params
    rf_params['max_depth'] = int(rf_params['max_depth'])
    rf_params['min_samples_leaf'] = int(rf_params['min_samples_leaf'])
    rf_params['n_estimators'] = int(rf_params['n_estimators'])
    rf = RandomForestRegressor(n_jobs=3, random_state=12345, **rf_params)

    # data
    X_train, y_train = preprocess_pipeline(data=data[ticker],
                                           indicators=indicators)
    X_test, y_test = preprocess_pipeline(data=test_data[ticker],
                                         indicators=test_indicators)
    
    # fit and score
    rf.fit(X_train.values, y_train.values)
    train_preds = rf.predict(X_train.values)
    test_preds = rf.predict(X_test.values)
    
    train_mse = rmse_ratio(y_train.values,
                           train_preds)
    train_mape = mean_absolute_percentage_error(y_train.values,
                                                train_preds)
    test_mse = rmse_ratio(y_test.values, 
                          test_preds)
    test_mape = mean_absolute_percentage_error(y_test.values,
                                               test_preds)
    
    # save model and results
    models[ticker] = {'predictions':test_preds, 
                      'train_mse':train_mse,
                      'train_mape':train_mape,
                      'test_mse':test_mse,
                      'test_mape':test_mape}
    print('For ticker %s, the train mape is %.3f and test mape is %.3f' % (
            ticker, train_mape, test_mape
        ))

# save results to csv for stack model training/testing
results = pd.DataFrame.from_dict(models, orient='index')
preds = pd.DataFrame([], columns=results.index)

for tick in tickers:
    preds[tick] = pd.Series(results.predictions[tick])

preds.index = test_data.index[11:]
preds = preds.reindex(test_data.index)

preds.round(4).to_csv('RF_predictions.csv') # saved all predictions

# for debugging, analysis
bad_models = results[(results['test_mape']/results['train_mape'] > 2) & (results['test_mape'] > 1.0)].index.tolist()
dict(params.loc[bad_models, 'param'].items())

# Visualize predictions for sanity check
import matplotlib.pyplot as plt

def chart(ticker):
    abc = pd.concat([preds[ticker].reset_index(drop=True),
                 test_data.loc['2016-01-16':, ticker]['CLOSE'].reset_index(drop=True)],
                axis=1)
    abc.plot(figsize=(12,9))
    plt.show()


chart('F_AD') # high accuracy
chart('F_DZ') # low accuracy
chart('F_LX') # funny


# Analysis of model meta (tree size, top features etc.)
abcd = results['F_LB'][0]
print([x.tree_.node_count/2 for x in abcd.estimators_])
cols = generate_features(data['F_AD']).columns
impts = abcd.feature_importances_
print(sorted(zip(cols, impts), key=lambda x: x[1], reverse=True))
