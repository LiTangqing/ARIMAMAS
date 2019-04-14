import statsmodels.tsa.api as sm
import pandas as pd
from itertools import product
import pickle
from multiprocessing import Pool

from tqdm import tqdm # python3

def grid_search(ticker):
    train = pd.read_pickle('./datasets/data_base_train.pkl')\
              .fillna(method='ffill')\
              .loc['2012-01-01':,(ticker, 'CLOSE')]
              
    params = list(product([0, 1, 2],
                      [1],                 
                      [1, 2]))
    seasonal_params = list(product([0, 1],
                                   [0, 1],
                                   [0, 1],
                                   [20]))
    best_aic = 10**7
    best_params = []
    for param in params:
        for param_s in seasonal_params:
            try:
                m = sm.SARIMAX(train, 
                               order=param,
                               seasonal_order=param_s,
                               trend='c'
                              ).fit(disp=False)
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_params = [param, param_s]
            except:
                continue
    print(ticker + ' Done!')
    if len(best_params) == 0:
        return 
    else:
        with open('SARIMA_' + ticker + '_params.txt', 'w') as f:
            f.write(str(best_params))
        return {'order':best_params[0], 'seasonal':best_params[1]}

if __name__ == "__main__":
    #train = pd.read_pickle('./datasets/data_base_train.pkl').fillna(method='ffill')
    #tickers = pd.read_csv('./csv_files/quantiacs_futures_details.csv')['Ticker'].tolist()
    tickers = ['F_AE', 'F_AH', 'F_AX', 'F_BO', 'F_BP', 'F_C',
               'F_CD', 'F_CF', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_EB', 'F_EC', 
             'F_ED', 'F_F', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY', 
             'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_JY', 'F_LU', 'F_LX', 'F_MD',
             'F_MP', 'F_ND', 'F_PQ', 'F_RF', 'F_RP', 'F_RR', 'F_RY', 'F_SF', 
             'F_SS', 'F_SX', 'F_TU', 'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_XX', 
             'F_YM', 'F_ZQ']#, 'F_VF', 'F_VT', 'F_VW']
    p = Pool(2)
    res = list(tqdm(p.imap(grid_search, tickers), total=len(tickers)))
    res = dict(tickers, res)
    pd.DataFrame.from_dict(res).to_csv('SARIMA_params.csv')
    print("##########DONE##########")
    #with open('SARIMA_params.pkl', 'wb') as f:
    #    pickle.dump(res, f)
        
