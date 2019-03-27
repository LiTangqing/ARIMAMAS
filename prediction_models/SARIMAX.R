setwd("~/GitHub/ARIMAMAS/prediction_models")

library(reticulate)
source_python("read_pickle.py")

setwd("~/GitHub/ARIMAMAS/data")
train_data = read_pickle_file("data_base_train.pkl")
test_data = read_pickle_file("data_base_test.pkl")

## indicators to use as xreg
train_indicators = read_pickle_file("data_indicators_base_train.pkl")
test_indicators = read_pickle_file("data_indicators_base_test.pkl")

library(zoo)

# All 88 futures
futures = list('F_AD','F_BO','F_BP','F_C','F_CC','F_CD',
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
           'F_VT','F_VW','F_GD','F_F')

pointer = 1

for (future in futures) {
  
  pointer = pointer + 1
}

# get close price and forward fill NaNs
x_train = na.locf(train_data[,(pointer*4)])
x_test = na.locf(test_data[,(pointer*4)])

# take log
x_train = log(x_train)
x_test = log(x_test)

period = 5 ## 5 trading days in a week

library(forecast)
#search for sarima models
min.sse.info = list(p=-1, d=-1, q=-1, P=-1, D=-1, Q=-1, 
                    sse=.Machine$double.xmax, res.p.value=100, model=NULL)
S = period

for (d in 1:1) for (D in 1:1) for (P in 0:2) for (Q in 0:2) for (p in 2:5) for (q in 2:5) {
  
  if (d+D+P+Q+p+q > 10) next
  
  model = tryCatch({
    arima(x_train,  order=c(p,d,q), seasonal=list(order=c(P,D,Q), period=S))
  }, warning = function(w) { message(w) }, 
  error   = function(e) { message(e); return(NULL) }, 
  finally = {})
  
  if (!is.null(model)) {
    fore.x = forecast(model, h=length(x_test))
    pred.x = as.numeric(fore.x$mean)
    sse = sum((pred.x-x_test)**2)
    box.test = Box.test(model$residuals, lag=log(length(model$residuals)))

    if (is.null(min.sse.info$model) || 
        sse < min.sse.info$sse ||
        (sse == min.sse.info$sse && 
         p+d+q+P+D+Q < min.sse.info$p + min.sse.info$d + min.sse.info$q + 
         min.sse.info$P + min.sse.info$D + min.sse.info$Q)){
      min.sse.info$sse = sse
      min.sse.info$p = p
      min.sse.info$d = d				
      min.sse.info$q = q
      min.sse.info$P = P
      min.sse.info$D = D				
      min.sse.info$Q = Q								
      min.sse.info$res.p.value = box.test$p.value			
      min.sse.info$model = model								
    }		
  }
}
