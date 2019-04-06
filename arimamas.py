import numpy as np
import pandas as pd

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model

from sklearn.linear_model import LinearRegression

# constants
MODEL_SAVED_DEST = "./prediction_models/LSTM_saved_models/"
LOOKBACK_LSTM = 30
LR_COE = "./prediction_models/LR_Model_Coefficients.csv"


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
    print(u'\U0001F37A')
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
    print("LSTM：done!")
    return np.vstack(predicted).reshape((-1))

def predict_lr(OPEN, HIGH, LOW, CLOSE, ticker_lists):
    coe_data = pd.read_csv(LR_COE)

    predictions = []
    for i, TICKER in enumerate(ticker_lists):
        # data for current ticker 
        data = np.array([OPEN[-1,i+1],HIGH[-1,i+1], LOW[-1,i+1], CLOSE[-1,i+1]]).reshape((-1))
        # get lr coefficient for current ticker
        coes = coe_data.loc[coe_data['Future']=="F_AD"].values.reshape((-1))[:5]
        # prediction = X*beta + intercept
        curr_pred = np.dot(coes[:4], data) + coes[4]

        predictions.append(curr_pred)
    return predictions



def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings,
                    USA_BC, USA_BI, USA_BOT, USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,
                    USA_CINF, USA_CP, USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
                    USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, USA_GBVL,
                    USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_LEI,
                    USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, USA_NFIB, USA_NFP, USA_NLTTF,
                    USA_NPP, USA_PFED, USA_PPIC, USA_RFMI, USA_RSEA, USA_RSM, USA_RSY,
                    USA_TVS, USA_UNR, USA_WINV):
    
    
    future_names = settings['markets'][1:] # remove cash
    n_futures = len(future_names)
    print("n_futures:", n_futures)

    # predict using lr
    lr_prediction = predict_lr(OPEN, HIGH, LOW, CLOSE, future_names)
    
    # predict using lstm
    lstm_prediction = predict_lstm(OPEN, HIGH, LOW, CLOSE, USA_BC, USA_BI, USA_BOT, 
                                   USA_CCPI, USA_CCR, USA_CF, USA_CFNAI,USA_CINF, USA_CP, 
                                   USA_CPI, USA_CPIC, USA_CPICM, USA_CU, USA_DUR,
                                   USA_DURET, USA_EXPX, USA_EXVOL, USA_FBI, USA_FRET, 
                                   USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, 
                                   USA_IPMOM, USA_LEI, USA_LFPR, USA_MP, USA_MPAY, USA_NAHB, 
                                   USA_NFIB, USA_NFP, USA_NLTTF, USA_NPP, USA_PFED, USA_PPIC, 
                                   USA_RFMI, USA_RSEA, USA_RSM, USA_RSY, USA_TVS, USA_UNR, USA_WINV, 
                                   future_names)


    return #weights, settings


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
     
    settings['lookback']= 31
    settings['budget']= 10**6
    settings['slippage']= 0.05

    settings['beginInSample'] = '20180101'
    settings['endInSample'] = '20190331'

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox

    results = quantiacsToolbox.runts(__file__)
