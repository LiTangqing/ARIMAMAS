setwd("~/GitHub/ARIMAMAS/prediction_models")

library(reticulate)
source_python("read_pickle.py")

setwd("~/GitHub/ARIMAMAS/data")
train_data = read_pickle_file("data_base_train.pkl")
test_data = read_pickle_file("data_base_test.pkl")
stack_data = read_pickle_file("data_stack_test.pkl")

## indicators to use as xreg
train_indicators = read_pickle_file("data_indicators_base_train.pkl")
test_indicators = read_pickle_file("data_indicators_base_test.pkl")
stack_indicators = read_pickle_file("data_indicators_stack_test.pkl")

train_indicators = na.locf(subset(train_indicators, select=c('USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY')))
test_indicators = na.locf(subset(test_indicators, select=c('USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY')))
stack_indicators = na.locf(subset(stack_indicators, select=c('USA_BC', 'USA_BOT', 'USA_CCR', 'USA_CF', 'USA_CPICM', 'USA_GPAY')))

train_indicators = train_indicators[2:(nrow(train_indicators)),]
test_and_stack_indicators = rbind(test_indicators, stack_indicators)

library(zoo)

# All 88 futures
futures = list('F_AD','F_AE','F_AH','F_AX','F_BC','F_BG','F_BO','F_BP',
               'F_C','F_CA','F_CC','F_CD','F_CF','F_CL','F_CT','F_DL',
               'F_DM','F_DT','F_DX','F_DZ','F_EB','F_EC','F_ED','F_ES',
               'F_F','F_FB','F_FC','F_FL','F_FM','F_FP','F_FV','F_FY',
               'F_GC','F_GD','F_GS','F_GX','F_HG','F_HO','F_HP','F_JY',
                'F_KC','F_LB','F_LC','F_LN','F_LQ','F_LR','F_LU','F_LX',
               'F_MD','F_MP','F_ND','F_NG','F_NQ','F_NR','F_NY','F_O',
               'F_OJ','F_PA','F_PL','F_PQ','F_RB','F_RF','F_RP','F_RR',
               'F_RU','F_RY','F_S','F_SB','F_SF','F_SH','F_SI','F_SM',
               'F_SS','F_SX','F_TR','F_TU','F_TY','F_UB','F_US','F_UZ',
               'F_VF','F_VT','F_VW','F_VX','F_W','F_XX','F_YM','F_ZQ')

# store predictions
results = setNames(data.frame(matrix(ncol = length(futures), nrow = 782)), futures)

# Add Date column

dates = c(test_dates, stack_dates)
results["DATE"] = dates

pointer = 1

for (future in futures) {
  
  # get close price and forward fill NaNs
  x_train = na.locf(train_data[,(pointer*4)])
  x_test = na.locf(test_data[,(pointer*4)])
  stack_test = na.locf(stack_data[,(pointer*4)])
  
  # take log
  x_train = log(x_train)
  x_test = log(x_test)
  x_test = c(NaN, x_test)   # first row has no close price
  stack_test = log(stack_test)
  stack_test = c(NaN, stack_test)   # first row has no close price
  
  x_and_stack_test = c(x_test, stack_test)
  
  period = 5 ## 5 trading days in a week
  S = period
  
  library(forecast)
  
  allpred = vector()
  
  test_dates = rownames(test_data)
  stack_dates = rownames(stack_data)
  
  batch_size = 5
  iterations = length(x_and_stack_test)/batch_size   
  
  # Build SARIMAX model iteratively to perform 1-step ahead forecast (for each trading week)
  for (i in 1:iterations) {
    train = c(x_train, x_and_stack_test[1:batch_size*(i-1)])
    # train model for all data up to simulated prediction day
    model = auto.arima(train, d = 1, max.p = 5, max.q =5 #, xreg = as.matrix(rbind(train_indicators, test_and_stack_indicators[1:batch_size*(i-1)]))
                       ,trace=FALSE)
    
    test = x_and_stack_test[(batch_size*(i-1) + 1):batch_size*i]
    pred = forecast(model, h = 5)  #, xreg = as.matrix(test_and_stack_indicators[(batch_size*(i-1) + 1):batch_size*i,]))    ## 1-step ahead forecast
    pred = as.numeric(pred$mean)
    allpred = c(allpred, exp(pred))
  }
  
  train = c(x_train, x_and_stack_test[1:(length(x_and_stack_test)-2)])
  model = auto.arima(train, d = 1, max.p = 5, max.q =5 #, xreg = as.matrix(rbind(train_indicators, test_and_stack_indicators[1:batch_size*(i-1)]))
                     ,trace=FALSE)
  test = x_and_stack_test[(length(x_and_stack_test)-1):(length(x_and_stack_test))]
  pred = forecast(model, h = 2)  #, xreg = as.matrix(test_and_stack_indicators[(batch_size*(i-1) + 1):batch_size*i,]))    ## 1-step ahead forecast
  pred = as.numeric(pred$mean)
  allpred = c(allpred, exp(pred))
  
  results[future] = allpred
  
  pointer = pointer + 1
}
