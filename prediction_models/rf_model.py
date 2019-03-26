# from __future__ import division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

__version__ = 0.1

# params
best_cols = ['F_C', 'F_CC', 'F_CT', 'F_DL', 'F_EB', 'F_ED', 'F_F', 'F_FB', 'F_FL',
             'F_FV', 'F_GC', 'F_GS', 'F_HP', 'F_KC', 'F_LC', 'F_ND', 'F_NG', 'F_NR',
             'F_O', 'F_OJ', 'F_PQ', 'F_RF', 'F_S', 'F_SB', 'F_SF', 'F_SM', 'F_SS',
             'F_TU', 'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VW', 'F_W', 'F_ZQ']

futures_filename = 'full_data_futures.csv' # all 88 futures closing prices
indicator_csv_filename = 'Indicators_training_interpolated.csv' # all indicators in same timeline

def preprocess(series, p=0, d=0, P=0, D=0, s=0, exog=None):
    train = pd.DataFrame()
    train['LABEL'] = series[1:]
    # engineered terms using simple ma, exp smoothing
    series = series.shift(1) # everything is lagged 1 as we get updated data everyday
    train['CLOSE'] = series
    train['MA5'] = series.rolling(5).mean()
    train['MA5_sd'] = series.rolling(5).std()
    train['MA20'] = series.rolling(20).mean()
    train['MA20_sd'] = series.rolling(20).std()
    train['EMA5'] = series.ewm(5).mean()
    train['EMA20'] = series.ewm(20).mean()
    if exog is not None:
        train[exog.columns] = exog
    
    for i in range(1,p+1):
        train['LAG_'+str(i)] = series.shift(i)
    if d > 0:
        train['DIFF'] = series.diff(d)
    if s > 0 and D > 0:
        train['sDIFF'] = series.diff(s+D)
        for i in range(1,P+1):
            train['sLAG_'+str(s+i)] = series.shift(s+i)
        
    train = train.fillna(method='ffill')\
                 .fillna(0)\
                 .reset_index(drop=True)
    train = np.asarray(train)
    return train[:,1:], train[:,0] # X, y

def score(model, X, y): # helper method to cross-validate
    tscv = TimeSeriesSplit(n_splits=6)
    splits = tscv.split(X)
    mse = []
    rsq = []
    for train_index, test_index in splits:
        model.fit(X[train_index], y[train_index])
        preds = rf.predict(X[test_index])
        mse.append(metrics.mean_squared_error(y[test_index], preds))
        rsq.append(metrics.r2_score(y[test_index], preds))
    return mse[1:], rsq[1:]

def visualize(ticker, rf, forecast_days=100, exog=None):
    X, y = preprocess(df[ticker], 10, 1, 5, 0, 20, exog)
    rf.fit(X[:-forecast_days], y[:-forecast_days])
    preds = rf.predict(X[-forecast_days:])
    print(metrics.r2_score(y[-forecast_days:], preds))
    fig = plt.figure(figsize=(12,9))
    plt.title(abc.loc[abc.Ticker == ticker, ['Ticker', 'Name']].values)
    plt.plot(preds, 'r-', label='preds')
    plt.plot(y[-forecast_days:], 'b-', label='true')
    plt.legend(loc='best')
    plt.show();
    
def fit(rf_params = {},
        preprocess_params = {'p':10, 'd':1, 'P':5, 'D':0, 'S':20, 'exog':None}
        cols = best_cols,
        df = None):
    "Fit an RF model to each future specified in cols. Returns a list of RF models."
    
    rf = RandomForestRegressor(**rf_params)
    models = []
    for col in cols:
        X, y = preprocess(df[col], **preprocess_params)
        rf.fit(X, y)
        models.append(rf)
    return models
    
if __name__ == "__main__":
    # default params
    rf = RandomForestRegressor(n_jobs=2,
                               random_state=12345,
                               n_estimators=150)
    # read in futures prices
    df = pd.read_csv(futures_filename, index_col='DATE',
                     parse_dates=True,
                     infer_datetime_format=True)
    # use data from 2010 onwards
    df = df['2010-01-01':].asfreq('D').asfreq('B').fillna(method='ffill')
    # read in economic indicators data
    indicators = pd.read_csv(indicator_csv_filename,
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
    # fit rf to each future and score
    res = []
    for col in best_cols:
        X, y = preprocess(df[col], 10, 1, 5, 0, 20, exog=indicators)
        mse, rsq = score(rf, X, y)
        res.append((col, mse, rsq))
        print('%s - MSE = %.2f, RSQ = %.4f' % (col, 
                                               np.mean(mse)/df[col].mean(),
                                               np.mean(rsq)))
    res = pd.DataFrame(res, columns=['Ticker', 'MSE', 'RSQ'])\
            .sort_values(by='RSQ', ascending=False).head(10)
    print(res.to_string())
        
        