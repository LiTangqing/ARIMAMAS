import numpy as np
import pickle
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Reshape, Flatten 
from keras.layers import LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model

from tqdm import tqdm
import os

from helper_functions import *

DATA_FOLDER = "../data/"
MODEL_SAVED_DEST = "./LSTM_saved_models/"
LOOKBACK = 30

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


TICKERS = [each[:-3] for each in os.listdir(MODEL_SAVED_DEST) if each.endswith(".h5") ]
df_list = []


for TICKER in tqdm(TICKERS):
	model = model = load_model(MODEL_SAVED_DEST + TICKER + '.h5')
	X_tr, y_tr, X_test, y_test, dates = prepare_train_test(TICKER ,train, test)
	X_to_pred = np.vstack((X_tr, X_test))
	y_true = np.concatenate((y_tr, y_test))

	to_pred_generator = TimeseriesGenerator(X_to_pred, y_true,
	                                    length=LOOKBACK, 
	                                    batch_size=1)

	y_pred = model.predict_generator(to_pred_generator)
	y_pred = y_pred.reshape(-1)

	y_scale_params = np.load(MODEL_SAVED_DEST+ TICKER + '_scale_params.npz')
	mu = y_scale_params['average_y']
	sd = y_scale_params['std_dev_y']

	y_pred = y_pred * sd + mu
	y_true = y_true * sd + mu

	dates = dates[31:]

	print("MAPE:", mean_absolute_percentage_error(y_pred, y_true[LOOKBACK:]))
	print("dates shape", dates.shape)
	print("y pred shape", y_pred.shape)

	predicted_df = pd.DataFrame({"DATE": pd.to_datetime(dates), TICKER: y_pred}).set_index("DATE")
	df_list.append(predicted_df)

# merge 
predicted = df_list[0]
for df in df_list[1:]:
	predicted = predicted.merge(df, how="outer", on="DATE")
predicted.to_csv("LSTM_predicted.csv")













