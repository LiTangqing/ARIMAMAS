import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Input, Reshape, Flatten,TimeDistributed
from keras.layers import LSTM, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import ConvLSTM2D

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities''' 
    future_names = settings['markets'][1:] # remove cash
    n_futures = len(future_names)
    print("n_futures:", n_futures)
    print("close shape:", CLOSE.shape)

    nMarkets=CLOSE.shape[1]
    print('CLOSE SHAPE:', CLOSE.shape)

    periodLonger=200
    periodShorter=40

    # Calculate Simple Moving Average (SMA)
    smaLongerPeriod=numpy.nansum(CLOSE[-periodLonger:,:],axis=0)/periodLonger
    smaShorterPeriod=numpy.nansum(CLOSE[-periodShorter:,:],axis=0)/periodShorter

    longEquity= smaShorterPeriod > smaLongerPeriod
    shortEquity= ~longEquity

    pos=numpy.zeros(nMarkets)
    pos[longEquity]=1
    pos[shortEquity]=-1

    weights = pos/numpy.nansum(abs(pos))

    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

     

    # Futures Contracts
    settings['markets'] = ['CASH','F_AD','F_BO','F_BP','F_C','F_CC',
                           'F_CD','F_CL','F_CT','F_DX','F_EC','F_ED',
                           'F_ES','F_FC','F_FV','F_GC','F_HG','F_HO',
                           'F_JY','F_KC','F_LB','F_LC','F_LN','F_MD',
                           'F_MP','F_NG','F_NQ','F_NR','F_O','F_OJ',
                           'F_PA','F_PL','F_RB','F_RU','F_S','F_SB',
                           'F_SF','F_SI','F_SM','F_TU','F_TY','F_US',
                           'F_W','F_XX','F_YM','F_AX','F_CA','F_DT',
                           'F_UB','F_UZ','F_GS','F_LX','F_SS','F_DL',
                           'F_ZQ','F_VX','F_AE','F_BG','F_BC','F_LU',
                           'F_DM','F_AH','F_CF','F_DZ','F_FB','F_FL',
                           'F_FM','F_FP','F_FY','F_GX','F_HP','F_LR',
                           'F_LQ','F_ND','F_NY','F_PQ','F_RR','F_RF',
                           'F_RP','F_RY','F_SH','F_SX','F_TR','F_EB',
                           'F_GD','F_F']
     
    settings['lookback']= 504
    settings['budget']= 10**6
    settings['slippage']= 0.05

    #settings['beginInSample'] = ''
    #settings['endInSample'] = ''

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
