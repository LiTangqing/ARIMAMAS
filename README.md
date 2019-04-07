BT4013 Project - ARIMAMAS
==========================

### Dir Layout
    .
    ├── data                        # compiled data sets in .pkl
    │   ├── data_base_train.pkl        
    │   └── ...                 
    ├── prediction_models           # individual prediction models                   
    │   ├── rf_model.py
    │   ├── csv_for_stacking
    │   │   ├── LSTM_predicted.csv
    │   │   └── ...  
    │   ├── LSTM_saved_models       # saved models & scaling parameters for each ticker
    │   └── ...      
    ├── weights_allocation          # weights allocation strategies for selected futures  
    │   └── ...    
    ├── arimamas.py                 # define myTradingSystem and mySettings
    └── README.md

### Dependencies
- numpy >= 1.15.0
- tensorflow 1.7.0
- keras 2.2.4

...

### Indicators to use
['USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY']

### Futures with less than 1 MAPE (trade these)
[F_AD, F_AE, F_AH, F_AX, F_BO, F_BP, F_C, F_CA, F_CD, F_CF, 
 F_DL, F_DM, F_DT, F_DX, F_EB, F_EC, F_ED, F_F, F_FC, F_FL, 
 F_FM, F_FP, F_FV, F_FY, F_GC, F_GD, F_GS, F_GX, F_JY, F_LU, 
 F_LX, F_MD, F_MP, F_ND, F_PQ, F_RF, F_RP, F_RR, F_RY, F_SF, 
 F_SS, F_SX, F_TU, F_TY, F_UB, F_US, F_UZ, F_VF, F_VT, F_VW, 
 F_XX, F_YM, F_ZQ]

### Models
#### 1. LSTM
Settings: 
- LOOKBACK = 30 (trained using days from past 30 days) 
- Each time make 1-step prediction
- Include all indicators as predictors

Model Architecture:
```python
model = Sequential()
model.add(LSTM(units=20, input_shape=(lookback, n_features),
               return_sequences=True, dropout=0.5))
model.add(LSTM(10, dropout=0.5)) 
model.add(Dense(units=20))
model.add(Dense(units=10))
model.add(Dense(units=dim_out))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='adam')
hist = model.fit_generator(generator, epochs=80, verbose=2)
```

Reproduce the model: 
- python ./prediction_models/lstm_for_stacking.py

Generate predictions for stacking:
- python ./prediction_models/LSTM_predict_for_stack.py
#### 2. Linear Regression
Settings: 
- Using .shift(1) to get 'LAG_OPEN', 'LAG_HIGH', 'LAG_LOW', 'LAG_CLOSE' as independent variables
- data_train_x & data_test_x consists of 'LAG_OPEN', 'LAG_HIGH', 'LAG_LOW', 'LAG_CLOSE'
- data_train_y & data_test_y consists of 'CLOSE'
- Refer to prediction_models/LR_Model_Coefficients.ipynb for the generated coefficients for each model
- Refer to prediction_models/csv_for_stacking/LR_Model_Predictions_(2016-2018).ipynb for the generated predictions for stacking

Model:
```python
lr = LinearRegression()
model = lr.fit(data_train_x, data_train_y)
y_pred = lr.predict(data_test_x)
```
