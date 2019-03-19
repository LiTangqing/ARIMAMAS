import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

 
from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Reshape, Flatten,TimeDistributed
from keras.layers import LSTM, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import ConvLSTM2D


def createAndTrain(DATE, CLOSE, settings):  
    _TS = 1 # timestamp  

    # read dates
    dates = DATE.tolist()[_TS:]
    
    # normalise price data  
    average = np.average(CLOSE[:-_TS, :], axis=0)
    std_dev = np.std(CLOSE[:-_TS, :], axis=0)
    prices = (CLOSE[:-_TS, :] - average) / std_dev
    # prices = normalize(CLOSE[:-_TS, :])
     
    returns = (CLOSE[_TS:, :] - CLOSE[:-_TS, :]) / CLOSE[:-_TS, :] 
    #return_data = CLOSE[:CLOSE.size-1]
    #return_data = (return_data - average) / std_dev 

    X = np.reshape(prices, (prices.shape[0], _TS, prices.shape[1]))
    ##########
    
    y = np.reshape(returns, (returns.shape[0], _TS, returns.shape[1]))
    

    dim_in = X.shape[2]
    dim_out = y.shape[2] # same as number of market

    y = returns
    dim_out = y.shape[1]
    
    kernel_size = 5
    filters = 64
    pool_size = 4

    # define model here
    model = Sequential()
     
    model.add(LSTM(units=20, input_shape=(_TS, X.shape[2]),
                   return_sequences=True, dropout=0.4))
    #model.add(Conv1D(filters=10, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(10, dropout=0.4)) 
    model.add(Dense(units=dim_out))
    model.add(Activation('linear'))

    #model.add(TimeDistributed(Dense(units=dim_out)))
    #model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam')
    
    print(model.summary())
    
    
    model.fit(X, y, epochs=300, batch_size=50, verbose=2)

    # plot hist
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()
    # #plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("hist.jpg")

    # variable to pass to settings
    
    settings['mean'] = average
    settings['std'] = std_dev
    settings['model'] = model
    return


def myTradingSystem(DATE, CLOSE, exposure, equity, settings):
    ''' This system uses mean reversion techniques to allocate capital into the desired equities '''
    lookBack = settings['lookback']
    if 'model' not in settings:
        createAndTrain(DATE[:lookBack - 2], CLOSE[:lookBack - 2,1:], settings)

    model = settings['model']
    average = settings['mean']
    std_dev = settings['std']

    testX = (CLOSE[lookBack-1,1:] - average) / std_dev
    testX = np.reshape(testX, (1, 1, testX.shape[0]))
    testY = model.predict(testX)
    predicted_returns = testY[0].flatten()
    
    if np.sum(predicted_returns) < 0:
        w_cash = -np.mean(predicted_returns)
    else:
        w_cash = np.mean(predicted_returns)

    w = predicted_returns/np.sum(abs(predicted_returns))
    w = np.append([w_cash], w)


    # pos = np.where(predicted_returns > 0, 1, -1)
    # pos = np.append([1], pos)
    #return pos, settings
     

    return w, settings

##### Do not change this function definition #####
def mySettings():
    ''' Define your trading system settings here '''
    settings = {}

    # Futures Contracts
    settings['markets'] = ['CASH','F_AD','F_BO','F_BP','F_C','F_CC','F_CD',
                           'F_CL','F_CT','F_DX','F_EC','F_ED','F_ES',
                           'F_FC','F_FV','F_GC','F_HG','F_HO','F_JY',
                           'F_KC','F_LB','F_LC','F_LN','F_MD','F_MP',
                           'F_NG','F_NQ','F_NR','F_O','F_OJ','F_PA',
                           'F_PL','F_RB','F_RU','F_S','F_SB','F_SF',
                           'F_SI','F_SM','F_TU','F_TY','F_US','F_W',
                           'F_XX','F_YM','F_AX','F_CA','F_DT','F_UB',
                           'F_UZ','F_GS','F_LX','F_SS','F_DL','F_ZQ',
                           'F_VX','F_AE','F_BG','F_BC','F_LU','F_DM',
                           'F_AH','F_CF','F_DZ','F_FB','F_FL','F_FM',
                           'F_FP','F_FY','F_GX','F_HP','F_LR','F_LQ',
                           'F_ND','F_NY','F_PQ','F_RR','F_RF','F_RP',
                           'F_RY','F_SH','F_SX','F_TR','F_EB','F_VF',
                           'F_VT','F_VW','F_GD','F_F']

    settings['slippage'] = 0.05
    settings['budget'] = 10 ** 6
    settings['lookback'] = 504
    settings['beginInSample'] = '20100101'
    settings['endInSample'] = '20181231'

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    from quantiacsToolbox import runts 

    np.random.seed(98274534)

    results = runts(__file__)

