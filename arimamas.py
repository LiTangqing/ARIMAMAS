#!/usr/bin/env python
# coding: utf-8

#from __future__ import print_function, division # for python 2

import numpy as np
import pandas as pd

#from keras.preprocessing.sequence import TimeseriesGenerator
#from keras.models import load_model

from sklearn.linear_model import LinearRegression
#from sklearn.externals.joblib import load
from helper_functions import generate_features
from portfolio_optimizer import slippage_costs

#import statsmodels.tsa.api as sm

#import lightgbm as lgb

# constants
MODEL_SAVED_DEST = "./prediction_models/LSTM_saved_models/"
LOOKBACK_LSTM = 30
LR_COE = "./prediction_models/LR_Model_Coefficients.csv"
LGBM_MODEL = "./prediction_models/LGBM_saved_models/"
STACKED_MODEL = "./prediction_models/Stacked_Model_Coefficients.csv"
RF_SAVED_DEST = "./prediction_models/RF_saved_models/"
SARIMA_SAVED_DEST = "./prediction_models/SARIMA_params.csv"

def predict_lstm(OPEN, HIGH, LOW, CLOSE, USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,
                    USA_CINF, USA_CP, USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
                    USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL,
                    USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_LEI,
                    USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, USA_NFIB, USA_NFP, USA_NLTTF,
                    USA_NPP, USA_PFED, USA_PPIC, USA_RFMI, USA_RSEA, USA_RSM, USA_RSY,
                    USA_TVS, USA_UNR, USA_WINV, ticker_lists):
    '''predict future close price using pretrained lstm model
       return: np array of predicted close price of shape (n_futures,) 
               the order is the same as settings['markets']
    '''

    print("===========================")
    print("LSTM is predicting...\nLSTM foresees a lot of work.\nLSTM works hard.")
    print("LSTM is slow but smart.\nBe like LSTM.")
    #print(u'\U0001F37A')
    predicted = []
    for i, TICKER in enumerate(ticker_lists):
        print("LSTM: working on "+TICKER+"...")
        # load model and scaling parameter
        model = load_model(MODEL_SAVED_DEST + TICKER+ '.h5')
        scale_params = np.load(MODEL_SAVED_DEST+ TICKER + '_scale_params.npz')
        mu_y = scale_params['average_y']
        sd_y = scale_params['std_dev_y']
        mu_X = scale_params['average_X']
        sd_X = scale_params['std_dev_X']
        
        # preprocess the data - concate, scale
        X = np.hstack((OPEN[:,i+1].reshape(-1,1), HIGH[:,i+1].reshape(-1,1), LOW[:,i+1].reshape(-1,1), CLOSE[:,i+1].reshape(-1,1),
                    USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF, USA_CFNAI, 
                    USA_CINF, USA_CP, USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR, USA_DURET,
                    USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX,
                    USA_IMVOL, USA_IP, USA_IPMOM, USA_LEI, USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, 
                    USA_NFIB, USA_NFP, USA_NLTTF, USA_NPP, USA_PFED, USA_PPIC, USA_RFMI, USA_RSEA, 
                    USA_RSM, USA_RSY, USA_TVS, USA_UNR, USA_WINV))

        X = (X - mu_X) / sd_X
        y_true = CLOSE[:,i+1] # dummy y - just to fill in 

        to_pred_generator = TimeseriesGenerator(X, y_true,
                                    length=30, 
                                    batch_size=1) 

        y_pred = model.predict_generator(to_pred_generator)
        y_pred = y_pred * sd_y + mu_y
        predicted.append(y_pred)
    print("LSTM: done!")
    return np.vstack(predicted).reshape((-1))

def predict_lr(OPEN, HIGH, LOW, CLOSE, ticker_lists):
    coe_data = pd.read_csv(LR_COE)

    predictions = []
    for i, TICKER in enumerate(ticker_lists):
        # data for current ticker 
        data = np.array([OPEN[-1,i+1],HIGH[-1,i+1], LOW[-1,i+1], CLOSE[-1,i+1]]).reshape((-1))
        # get lr coefficient for current ticker
        coes = coe_data.loc[coe_data['Future']==TICKER].values.reshape((-1))[:5]
        # prediction = X*beta + intercept
        curr_pred = np.dot(coes[:4], data) + coes[4]

        predictions.append(curr_pred)
    return predictions

def predict_lgbm(OPEN, HIGH, LOW, CLOSE, USA_BC, USA_BOT, USA_CCR, USA_CF, USA_CPICM, USA_GPAY, ticker_lists):
    predictions = []
    for i, TICKER in enumerate(ticker_lists):
        filename = LGBM_MODEL + TICKER + '.txt'
        with open(filename, 'r') as f:
            model_str = f.read()
        bst = lgb.Booster({'model_str':model_str})
        dat = np.array([OPEN[-1,i+1], HIGH[-1,i+1], LOW[-1,i+1], CLOSE[-1,i+1],
                        USA_BC[-1,:], USA_BOT[-1,:], USA_CCR[-1,:],
                        USA_CF[-1,:], USA_CPICM[-1,:], USA_GPAY[-1,:]]).reshape((1,10))
        curr_pred = bst.predict(dat)[0]

        predictions.append(curr_pred)
    return np.vstack(predictions).reshape((-1))

def predict_rf(OPEN, HIGH, LOW, CLOSE,
               USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,
               USA_CINF, USA_CP, USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
               USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL,
               USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_LEI,
               USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, USA_NFIB, USA_NFP, USA_NLTTF,
               USA_NPP, USA_PFED, USA_PPIC, USA_RFMI, USA_RSEA, USA_RSM, USA_RSY,
               USA_TVS, USA_UNR, USA_WINV, ticker_lists):
    '''
    Uses prices and indicators information to predict with Random Forest.
    
    Engineers features for prediction using helper_functions script.
    '''
    indicators = pd.DataFrame(
            np.hstack((USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF,
                       USA_CFNAI, USA_CINF, USA_CP, USA_CPI, USA_CPIC,
                       USA_CPICM, USA_CU, USA_DUR, USA_DURET, USA_EXPX,
                       USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL, USA_GPAY, 
                       USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM,
                       USA_LEI, USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, 
                       USA_NFIB, USA_NFP, USA_NLTTF, USA_NPP, USA_PFED, 
                       USA_PPIC, USA_RFMI, USA_RSEA, USA_RSM, USA_RSY,
                       USA_TVS, USA_UNR, USA_WINV))[-40:,:]
    )
    predicted = []
    for i, TICKER in enumerate(ticker_lists):
        print("RF: working on "+TICKER+"...")
        # load model from joblib
        model = load(RF_SAVED_DEST + 'RF_' + TICKER + '.joblib')
            
        # preprocess the data - concat and engineer features
        data = pd.DataFrame({'OPEN':OPEN[-40:,i+1],
                             'HIGH':HIGH[-40:,i+1],
                             'LOW':LOW[-40:,i+1],
                             'CLOSE':CLOSE[-40:,i+1]})
        data = generate_features(data, p=4, d=1, P=2, D=0, s=5, ma=[5,40])
        data = pd.concat([data, indicators],
                          axis=1).dropna()
        
        # predict and save results
        y_pred = model.predict(data.tail(1).values)
        predicted.append(y_pred)
    print("RF done!")
    return np.vstack(predicted).reshape((-1))

def predict_sarima(CLOSE, ticker_lists):
    """Make 1-step forecast of CLOSE prices in tickers.
    
    Uses a csv of saved order & seasonal order for model specification."""
    predicted = []
    # Load grid-searched params from csv
    params = pd.read_csv(SARIMA_SAVED_DEST, index_col=0)
    
    print("Super ARIMAMA getting to work!")
    for i, TICKER in enumerate(ticker_lists):
        print("SARIMA: working on "+TICKER+"...")
        
        # convert str params to tuple
        order = eval(params.loc[TICKER, 'order']) 
        seasonal_order = eval(params.loc[TICKER,'seasonal'])
        
        # fit model based on params
        model = sm.SARIMAX(CLOSE[-120:, i+1],
                           order=order,
                           seasonal_order=seasonal_order,
                           trend='c', 
                           enforce_stationarity=False, # bypass any errors
                           enforce_invertibility=False)\
                    .fit(disp=False)
        
        # predict and save results
        y_pred = model.forecast(1)
        predicted.append(y_pred)
        
    print("SARIMA done!")
    return np.vstack(predicted).reshape((-1))

def predict_stacked(LGBM, LSTM, RF, LR, SARIMA, ticker_lists):
    coe_data = pd.read_csv(STACKED_MODEL)

    predictions = []
    for i, TICKER in enumerate(ticker_lists):
        # data for current ticker 
        data = np.array([LGBM[i], LSTM[i], RF[i], LR[i], SARIMA[i]])
        
        # get base model coefficients for current ticker
        coes = coe_data.loc[coe_data['Future']==TICKER].values.reshape((-1))[:6]
        
        # prediction = X*beta + intercept
        curr_pred = np.dot(coes[:5], data) + coes[5]

        predictions.append(curr_pred)
    return np.vstack(predictions).reshape((-1))

def stacked_momentum(OPEN, HIGH, LOW, CLOSE, preds, settings):
    slip = slippage_costs(HIGH[-1,1:51],
                          LOW[-1,1:51],
                          CLOSE[-2,1:51],
                          np.ones((len(settings['mape1']),)),
                          settings['slippage'])

    pos = (preds - CLOSE[-1,1:51])/CLOSE[-1,1:51]
    pos = np.where((pos-slip < 0.01) & (pos+slip > -0.01), 0, pos) / np.nansum(np.abs(pos))
    pos = np.insert(pos, 0, np.nanmedian(np.abs(pos))) # give some weight to cash to reduce position changes
    return np.nan_to_num(pos)

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings,
                    USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,
                    USA_CINF, USA_CP, USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
                    USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL,
                    USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_LEI,
                    USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, USA_NFIB, USA_NFP, USA_NLTTF,
                    USA_NPP, USA_PFED, USA_PPIC, USA_RFMI, USA_RSEA, USA_RSM, USA_RSY,
                    USA_TVS, USA_UNR, USA_WINV):
    
    
    future_names = settings['mape1'] 
    n_futures = len(future_names)
    
    
#     # predict using lgbm
#     lgbm_prediction = predict_lgbm(OPEN, HIGH, LOW, CLOSE,
#                                    USA_BC, USA_BOT, USA_CCR, USA_CF, 
#                                    USA_CPICM, USA_GPAY, future_names)

#     # predict using lr
#     lr_prediction = predict_lr(OPEN, HIGH, LOW, CLOSE, future_names)
    
#    # predict using lstm
#    lstm_prediction = predict_lstm(OPEN, HIGH, LOW, CLOSE, USA_BC, USA_BI, USA_BOT, 
#                                   USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,USA_CINF, USA_CP, 
#                                   USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
#                                   USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, 
#                                   USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, 
#                                   USA_IPMOM, USA_LEI, USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, 
#                                   USA_NFIB, USA_NFP, USA_NLTTF, USA_NPP, USA_PFED, USA_PPIC, 
#                                   USA_RFMI, USA_RSEA, USA_RSM, USA_RSY, USA_TVS, USA_UNR, USA_WINV, 
#                                   future_names)
    
#     # predict using rf
#     rf_prediction = predict_rf(OPEN, HIGH, LOW, CLOSE, USA_BC, USA_BI, USA_BOT, 
#                                    USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,USA_CINF, USA_CP, 
#                                    USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
#                                    USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, 
#                                    USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, 
#                                    USA_IPMOM, USA_LEI, USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, 
#                                    USA_NFIB, USA_NFP, USA_NLTTF, USA_NPP, USA_PFED, USA_PPIC, 
#                                    USA_RFMI, USA_RSEA, USA_RSM, USA_RSY, USA_TVS, USA_UNR, USA_WINV, 
#                                    future_names)
    
#     # predict using sarima
#     sarima_prediction = predict_sarima(CLOSE, future_names)

    date = pd.to_datetime(DATE[-1], format="%Y%m%d")
    lgbm_prediction = settings['lgbm'].loc[date, :].values
    lstm_prediction = settings['lstm'].loc[date, :].values
    rf_prediction = settings['rf'].loc[date, :].values
    lr_prediction = settings['lr'].loc[date, :].values
    sarima_prediction = settings['sarima'].loc[date, :].values
    
    # predict using stacked model
    stacked_prediction = predict_stacked(lgbm_prediction,
                                         lstm_prediction,
                                         rf_prediction,
                                         lr_prediction,
                                         sarima_prediction,
                                         future_names)
    
    # optimize weight allocation strategy
    weights = stacked_momentum(OPEN, HIGH, LOW, CLOSE, stacked_prediction, settings) 
    weights = np.concatenate((weights, np.zeros(38)))
    
    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}
    mape1 = settings['mape1'] = ['F_AD', 'F_AE', 'F_AH', 'F_AX', 'F_BO', 'F_BP', 'F_C', 'F_CA',
             'F_CD', 'F_CF', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_EB', 'F_EC', 
             'F_ED', 'F_F', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY', 
             'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_JY', 'F_LU', 'F_LX', 'F_MD',
             'F_MP', 'F_ND', 'F_PQ', 'F_RF', 'F_RP', 'F_RR', 'F_RY', 'F_SF', 
             'F_SS', 'F_SX', 'F_TU', 'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_XX', 
             'F_YM', 'F_ZQ'] 
    
    non_mape1 = ['F_BC','F_BG', 'F_CC' ,'F_CL','F_CT', 'F_DZ', 'F_ES', 'F_FB',
    'F_HG','F_HO','F_HP','F_KC', 'F_LB','F_LC','F_LN','F_LQ','F_LR', 'F_NG','F_NQ','F_NR','F_NY', 
    'F_O','F_OJ', 'F_PA','F_PL', 'F_RB' ,'F_RU', 'F_S','F_SB','F_SH','F_SI','F_SM', 
    'F_TR', 'F_VF','F_VT','F_VW','F_VX', 'F_W']
    
    # Futures Contracts
    # all 88 
    settings['markets'] = ['CASH'] + mape1 + non_mape1 
    print('Number of futures:', len(settings['markets']))
    
    settings['beginInSample'] = '20170119'
    settings['endInSample'] = '20190331' 

    # settings['beginInSample'] = '20170119'
    # settings['endInSample'] = '20190408' 

    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05
    
    # read in necessary data
    index = pd.DatetimeIndex(start='2019-01-02',
                             end='2019-04-08',
                             freq='B')
    root = './prediction_models/csv_for_stacking/'
    
    lgbm = pd.read_csv(root + 'LGBM_Model_Predictions_(2019)V2.csv', index_col=0)
    lgbm.index = index
    settings['lgbm'] = lgbm.reindex(columns=mape1)
    
    lstm = pd.read_csv(root + 'LSTM_Model_Predictions_(2019)V2.csv', index_col=0)
    lstm.index = index
    settings['lstm'] = lstm.reindex(columns=mape1)
    
    rf = pd.read_csv(root + 'RF_Model_Predictions_(2019)V2.csv', index_col=0)
    rf.index = index
    settings['rf'] = rf.reindex(columns=mape1)
    
    lr = pd.read_csv(root + 'LR_Model_Predictions_(2019)V2.csv', index_col=0)
    lr.index = index
    settings['lr'] = lr.reindex(columns=mape1)
    
    sarima = pd.read_csv(root + 'SARIMA_Model_Predictions_(2019)V2.csv', index_col=0)
    sarima.index = index
    settings['sarima'] = sarima.reindex(columns=mape1)
                                
    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
