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

### Read from .pkl files
``` python
data = pickle.load(open(DATA_PATH, "rb" ), encoding='latin1') # for python3
data = pickle.load(open(DATA_PATH, "rb" )) # for python2
```
### Indicators to use
['USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY']

### Models
#### 1. LSTM
Trained using days from past 30 days, each time make 1-step prediction.

### TODO
- finish individual models & make prediction
- stacking
- research & work on weights allocation strategies
