import numpy as np
import pickle
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Reshape, Flatten 
from keras.layers import LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

from tqdm import tqdm
import time

from helper_functions import *

DATA_FOLDER = "../data/"
MODEL_SAVED_DEST = "./LSTM_saved_models/"

# constants
LOOKBACK = 30

import os

futures = pd.read_csv(DATA_FOLDER + "quantiacs_futures_details.csv")
tickers = futures['Ticker']

def read_from_pkl(PATH):
    data = pickle.load(open(DATA_FOLDER + PATH, "rb" ), encoding='latin1')
    return data

train = pd.concat([read_from_pkl("data_base_train.pkl" ),
                   read_from_pkl("data_base_test.pkl") ]) 
train_idc = pd.concat([read_from_pkl("data_indicators_base_train.pkl"),
                      read_from_pkl("data_indicators_base_test.pkl") ])
test = read_from_pkl('data_stack_test.pkl')
test_idc = read_from_pkl('data_indicators_stack_test.pkl')



def preprocess_train(price_data, indicator):
    price_data = price_data.loc[price_data.index >= '1994-01-01']
    start_index = price_data['CLOSE'].first_valid_index()
    price_data = price_data.loc[price_data.index >= start_index]
    print("price data start from: ", start_index)
    print("merge with indicators ...")
    train = price_data.join(indicator, on='DATE')
    print("fill NAs ...")
    train = train.fillna(method="bfill").reset_index()
    #train.rename(columns = {'DATE':'ds','CLOSE':'y'},inplace=True)
    print("Done")
    return train

def preprocess_test(price_data, indicator):
    #price_data = price_data.loc[price_data.index >= '1994-01-01']
    #start_index = price_data['CLOSE'].first_valid_index()
    #price_data = price_data.loc[price_data.index >= start_index]
    #print("price data start from: ", start_index)
    print("merge with indicators ...")
    test = price_data.join(indicator, on='DATE')
    print("fill NAs ...")
    test = test.fillna(method="bfill").reset_index()
    #test.rename(columns = {'DATE':'ds','CLOSE':'y'},inplace=True)
    print("Done")
    return test

def prepare_train_test(ticker_name, train, test):
    
    # filter data for such ticker & preprocess
    train = preprocess_train(train[ticker_name], train_idc)
    test = preprocess_test(test[ticker_name], test_idc)
    
    # split X and y
    X_tr_date = train['DATE']
    X_tr = train.drop(['DATE'], axis=1).values
    y_tr = train.loc[:,"CLOSE"].values

    X_tr = X_tr[0: X_tr.shape[0]-1]
    y_tr = y_tr[1:] # 1 day ahead
    
    # scale X and y 
    average_X = np.average(X_tr, axis=0)
    std_dev_X = np.std(X_tr, axis=0)

    average_y = np.average(y_tr)
    std_dev_y = np.std(y_tr)
    
    X_tr = (X_tr - average_X)/std_dev_X
    y_tr = (y_tr - average_y)/std_dev_y
    
    # scale the test set using the same params
    X_test_date = test['DATE']
    X_test = test.drop(['DATE'], axis=1).values
    X_test = (X_test - average_X)/std_dev_X
    
    y_test = test.loc[:,"CLOSE"].values
    y_test = (y_test - average_y)/std_dev_y
    
    # save scale param into npz 
    np.savez(MODEL_SAVED_DEST + ticker_name + '_scale_params.npz', 
             average_X=average_X, std_dev_X = std_dev_X,
             average_y=average_y, std_dev_y = std_dev_y) 
    print('X_tr shape:', X_tr.shape)
    print('y_tr shape:', y_tr.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    
    return X_tr, y_tr, X_test, y_test, pd.concat([X_tr_date, X_test_date])
    


def train_lstm(X_tr, y_tr, lookback=LOOKBACK):
    
    n_features = X_tr.shape[1]
    dim_out = 1  
    
    generator = TimeseriesGenerator(X_tr, y_tr, 
                                    length=LOOKBACK, 
                                    batch_size=32)
 
    # define model here
    model = Sequential()

    model.add(LSTM(units=20, input_shape=(lookback, n_features ),
                   return_sequences=True, dropout=0.5))
    model.add(LSTM(10, dropout=0.5)) 
    model.add(Dense(units=20))
    model.add(Dense(units=10))
    model.add(Dense(units=dim_out))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')
    
    print(model.summary())
    #model.fit(X, y, epochs=200, batch_size=32, verbose=2)
    model.fit_generator(generator,epochs=80, verbose=2)
    return model



for ticker in tqdm(tickers):
    print("===========================")
    print("training for ticker:", ticker)
    
    X_tr, y_tr, X_test, y_test, dates = prepare_train_test(ticker ,train, test)
    
    model = train_lstm(X_tr, y_tr)
    model.save(MODEL_SAVED_DEST + ticker + ".h5")
    
#     X_to_pred = np.vstack((X_tr, X_test))
#     y_true = np.concatenate((y_tr, y_test))
#     to_pred_generator = TimeseriesGenerator(X_to_pred, y_true,
#                                     length=LOOKBACK, 
#                                     batch_size=1)
#     y_pred = model.predict_generator(to_pred_generator)
#     y_pred = y_pred.reshape(-1)
    
#     y_scale_params = np.load(MODEL_SAVED_DEST+ ticker + '_scale_params.npz')
#     mu = y_scale_params['average_y']
#     sd = y_scale_params['std_dev_y']

#     y_pred = y_pred * sd + mu
#     y_true = y_true * sd + mu
    
#     print("MAPE:", mean_absolute_percentage_error(y_pred, y_true[LOOKBACK:]))
    
#     print("dates shape", dates.shape)
#     print("y pred shape", y_pred.shape)
    

# y_pred = model.predict_generator(to_pred_generator)
# y_pred = y_pred.reshape(-1)
# print("MAPE:", mean_absolute_percentage_error(y_pred, y_true[LOOKBACK:]))
    
# print("dates shape", dates.shape)
# print("y pred shape", y_pred.shape)

