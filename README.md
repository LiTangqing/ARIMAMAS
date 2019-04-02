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
    ├── ClosePriceEvaluation.py  
    └── README.md

### Dependencies
- numpy >= 1.15.0
- tensorflow 1.7.0
...

### Indicators to use
['USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY']

### Models
#### 1. LSTM
Settings: 
- LOOKBACK = 30 (trained using days from past 30 days) 
- Each time make 1-step prediction
- Include all indicators as predictors

Model Architecture:
```python
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
model.fit_generator(generator,epochs=80, verbose=2)
```

### TODO
- ~~finish individual models & make prediction~~
- stacking
- research & work on weights allocation strategies
#### TO ADD TO README:
- add instructions for reproduce and use each of the model 
- add information on how indicators are picked 
- add training & tuning strategy
