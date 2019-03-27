BT4013 Project - ARIMAMAS
==========================

### Dir Layout
    .
    ├── data                      # compiled data sets in .pkl
    │   ├── data_base_train.pkl        
    │   └── ...                 
    ├── prediction_models         # individual prediction models                   
    │   ├── rf_model.py
    │   └── ...      
    ├── weights_allocation        # weights allocation strategies for selected futures (based on prediction result)
    │   └── ...    
    ├── ClosePriceEvaluation.py  
    └── README.md

### Dependencies
- numpy
- tensorflow 1.7.0 
...

### Read from .pkl files 
``` python
data = pickle.load(open(DATA_PATH, "rb" ), encoding='latin1') # for python3 
data = pickle.load(open(DATA_PATH, "rb" )) # for python2 
```

### Indicators to use
['USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY']

### TODO 
- finish individual models & make prediction
- stacking 
- research & work on weights allocation strategies 
