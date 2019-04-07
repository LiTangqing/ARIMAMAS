# IMPORTS
import numpy as np
import pandas as pd

__version__ = '2.0'

def generate_features(df, p=0, d=0, P=0, D=0, s=0, ma=[5, 40]):
    """Given a pandas DataFrame, generate moving-average, price-action and lagged features.
    
    Params:
    * p: AR(p) order (int)
    * d: Simple Differencing order (int)
    * P: seasonal AR(p) order (int)
    * D: Seasonal Differencing order (int)
    * s: seasonality period (int)
    * ma: Moving average periods to compute (list of int)
    
    Returns:
    * train (Pandas DataFrame)
        Make sure to drop any NA values created from lagged features.
    """
    train = df.copy() # deep copy
    
    # Moving averages (for trend & volatility)
    for ma_num in ma:
        string_num = str(ma_num)
        train['SMA'+string_num] = train.rolling(ma_num, min_periods=2).mean()['CLOSE']
        train['SMA'+string_num+'_std'] = train.rolling(ma_num, min_periods=2).std()['CLOSE']
        train['EMA'+string_num] = train.ewm(ma_num).mean()['CLOSE']
        train['EMA'+string_num+'_std'] = train.ewm(ma_num).std()['CLOSE']
    
    # Price action features
    train['High_minus_low'] = train['HIGH'] - train['LOW']
    train['Close_minus_open'] = train['CLOSE'] - train['OPEN']
    train['Resistance'] = train.rolling(20, min_periods=2).max()['HIGH']
    train['Support'] = train.rolling(20, min_periods=2).min()['LOW']
    
    # Autoregressive TS features
    if d > 0:
        train['DIFF'] = train['CLOSE'].diff(d)
    if s > 0 and D > 0:
        train['sDIFF'] = train['CLOSE'].diff(s*D)
    for i in range(1,p+1):
        train['LAG_'+str(i)] = train['CLOSE'].shift(i)
    for i in range(1,P+1):
        train['sLAG_'+str(s*i)] = train['CLOSE'].shift(s*i)
        
    return train

def preprocess_pipeline(data, indicators):
    """For each ticker data, concat with indicators and spit out X, y."""
    train = data.copy()
    train.loc[:, 'TARGET'] = train['CLOSE'].shift(-1) # use today predict tmr
    train = generate_features(train, 
                              p=4, d=1, P=2, D=0, s=5, ma=[5,40])
    train = pd.concat([train.reset_index(drop=True),
                       indicators.reset_index(drop=True)],
                      axis=1).dropna()
    return train.drop(columns=['TARGET']), train['TARGET']

def rmse_ratio(y_true, y_preds):
    """Computes sqrt of mean of squared error / y_true."""
    se = np.square(y_preds - y_true) / y_true
    return np.mean(se) ** 0.5

def mean_absolute_percentage_error(y_true, y_preds):
    """Computes mean of abs error / y_true. Returns percentage value (i.e. *100)."""
    ape = np.absolute(y_preds - y_true) / y_true
    return np.mean(ape) * 100