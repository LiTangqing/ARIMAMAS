setwd("C:/Users/Cheong/Desktop/Yi-Wei-Now/BT4013/Project/Trendfollowing-Sample-Strategy/tickerData")

all_models = data.frame(matrix(nrow=3, ncol=89))
vec = vector()
count = 1
for (i in list.files()) {
  temp = read.table(i, header=TRUE, sep = ',')
  name = strsplit(i, ".", fixed=TRUE)[[1]][1]
  vec = c(vec, name)
  
  price = temp[,5]
  log.price = log(price)
  
  # grid search for p in 0:5, d in 0:1, q in 0:5
  
  min.aic.info = list(p=-1, d=-1, q=-1, sse=.Machine$double.xmax, res.p.value=100, model=NULL)
  
  for (p in 0:5) for (d in 0:1) for (q in 0:5) {
    model = tryCatch({
      stats::arima(log.price, order=c(p,d,q))#, include.mean=FALSE) 
    }, warning = function(w) {message(w)},
    error = function(e) {message(e); return(NULL)},
    finally = {}
    )
    
    if (is.null(model)) {
      next
    }
    
    if (is.null(min.aic.info$model) || 
        model$aic < (min.aic.info$model)$aic ||
        (model$aic == (min.aic.info$model)$aic && p+q < min.aic.info$p+min.aic.info$q)){
      min.aic.info$p = p
      min.aic.info$d = d
      min.aic.info$q = q
      min.aic.info$model = model			
    }		
  }
  
  all_models[1,count] = min.aic.info$p 
  all_models[2,count] = min.aic.info$d 
  all_models[3,count] = min.aic.info$q
  
  count = count + 1
}
colnames(all_models) = vec
rownames(all_models) = c("p","d","q")

setwd("C:/Users/Cheong/Desktop/Yi-Wei-Now/BT4013/Project")
write.csv(all_models, file = "PDQs.csv")
