rm(list=objects())
install.packages("tidyverse")
install.packages("tibble")
install.packages("parallel")
install.packages("tsoutliers")
install.packages("lmtest")
install.packages("lattice")
install.packages("grid")
install.packages("DMwR")
install.packages("ROI")
install.packages("rpart")
install.packages("tseries")
install.packages("TTR")
install.packages("xts")
install.packages("randomForest")
install.packages("DescTools")
install.packages("rpart.plot")
library(graphics)
library(tibble)
library(tidyr)
library(base)
library(dplyr)
library(readr)
library(parallel)
library(lmtest)
library(lattice)
library(grid)
library(DMwR)
library(car)
library(rpart)
library(ROI)
library(class)
library(tseries)
library(xts)
library(quantmod)
library(TTR)
library(randomForest)
library(DescTools)
library(rpart.plot)
setwd("C:/Users/samer/Documents/Master/Apprentissage statistique/TP1")
options("scipen"=100, "digits"=10)
data(iris)
head(iris)
summary(iris)

#Séparer les données en deux train et test set.
train = iris[c(1:30,51:80,101:130),1:5]
test = iris[c(31:50,81:100,131:150),1:5]
pred = knn(train[,1:4], test[,1:4], train[,5], k = 7)
# display the confusion matrix
table(pred,test[,5])
#Ces résultats montrent que le clasifieur kNN avec k = 3 conduit à une erreur de classification très faible (1.67%) dans ce problème

#validation croisée
# 5-fold cross-validation to select k
# from the set {1,...,10}
fold = sample(rep(1:5,each=18)) # creation des groupes B_v
cvpred = matrix(NA,nrow=90,ncol=10) # initialisation de la matrice
# des prédicteurs
for (k in 1:10)
  for (v in 1:5)
  {
    sample1 = train[which(fold!=v),1:4]
    sample2 = train[which(fold==v),1:4]
    class1 = train[which(fold!=v),5]
    cvpred[which(fold==v),k] = knn(sample1,sample2,class1,k=k)
  }
class = as.numeric(train[,5])
# display misclassification rates for k=1:10
apply(cvpred,2,function(x) sum(class!=x)) # calcule l'erreur de classif
#normalement 7 le meilleur K

#Predicting stock market returns

data(GSPC)
head(GSPC)

#Implementing the simpled indicator
T.ind = function(quotes, tgt.margin = 0.025, n.days = 10) {
  v = apply(HLC(quotes), 1, mean)
  r = matrix(NA, ncol = n.days, nrow = NROW(quotes))
  for (x in 1:n.days) r[, x] = Next(Delt(v, k = x), x)
  x = apply(r, 1, function(x) sum(x[x > tgt.margin | x <
                                      -tgt.margin]))
  if (is.xts(quotes))
    xts(x, time(quotes))
  else x
}


candleChart(last(GSPC, "3 months"), theme = "white", TA = NULL)
avgPrice = function(p) apply(HLC(p), 1, mean)
medianPrice=function(p) apply(HLC(p),1,median)
addAvgPrice = newTA(FUN = avgPrice, col = 1, legend = "AvgPrice")
addMedianPrice=newTA(FUN=medianPrice,col="green", legend="MedianPrice")
addT.ind = newTA(FUN = T.ind, col = "red", legend = "tgtRet")
get.current.chob<-function(){quantmod:::get.current.chob()}
candleChart(last(GSPC, "3 months"), theme = "white", TA = "addAvgPrice()")
candleChart(last(GSPC, "3 months"), theme = "white", TA = "addAvgPrice(on=1)")
candleChart(last(GSPC, "3 months"), theme = "white", TA = "addT.ind();addAvgPrice(on=1);addMedianPrice(on=1)")


#Predictors
myATR = function(x) ATR(HLC(x))[, "atr"]
mySMI = function(x) SMI(HLC(x))[, "SMI"]
myADX = function(x) ADX(HLC(x))[, "ADX"]
myAroon = function(x) aroon(x[, c("High", "Low")])$oscillator
myBB = function(x) BBands(HLC(x))[, "pctB"]
myChaikinVol = function(x) Delt(chaikinVolatility(x[, c("High","Low")]))[, 1]
myCLV = function(x) EMA(CLV(HLC(x)))[, 1]
myEMV = function(x) EMV(x[, c("High", "Low")], x[, "Volume"])[,2]
myMACD = function(x) MACD(Cl(x))[, 2]
myMFI = function(x) MFI(x[, c("High", "Low", "Close")],x[, "Volume"])
mySAR = function(x) SAR(x[, c("High", "Close")])[, 1]
myVolat = function(x) volatility(OHLC(x), calc = "garman")[,1]

data.model = specifyModel(T.ind(GSPC) ~ Delt(Cl(GSPC),k=1:10) +
                            myATR(GSPC) + mySMI(GSPC) + myADX(GSPC) + myAroon(GSPC) +
                            myBB(GSPC) + myChaikinVol(GSPC) + myCLV(GSPC) +
                            CMO(Cl(GSPC)) + EMA(Delt(Cl(GSPC))) + myEMV(GSPC) +
                            myVolat(GSPC) + myMACD(GSPC) + myMFI(GSPC) + RSI(Cl(GSPC)) +
                            mySAR(GSPC) + runMean(Cl(GSPC)) + runSD(Cl(GSPC)))
set.seed(1234)

rf = buildModel(data.model,method="randomForest",
                training.per=c(start(GSPC),index(GSPC["1999-12-31"])),
                ntree=50, importance=T)

#Checking the importance of the variables
varImpPlot(rf@fitted.model, type = 1)

#nouveau modèle
data.model = specifyModel(T.ind(GSPC) ~ myATR(GSPC) + myADX(GSPC) + myCLV(GSPC) +
                            myEMV(GSPC) + myVolat(GSPC) + myMACD(GSPC) + mySAR(GSPC) + 
                            runMean(Cl(GSPC)))

                    
#Predicting model
Tdata.train = as.data.frame(modelData(data.model,
                                      data.window=c("1970-01-02","1999-12-31")))
Tdata.eval = na.omit(as.data.frame(modelData(data.model,
                                             data.window=c("2000-01-01","2009-09-15"))))

Tdata.train[,1] = trading.signals(Tdata.train[,1],0.1,-0.1)
names(Tdata.train)[1] = "signal"
summary(Tdata.train)


Tdata.eval[,1] = trading.signals(Tdata.eval[,1],0.1,-0.1)
names(Tdata.eval)[1] = "signal"
summary(Tdata.eval)

x<-vector()
for (i in 1:100)
{
#validation croisée
# 6-fold cross-validation to select k
# from the set {1,...,10}
fold = sample(rep(1:6,each=1257)) # creation des groupes B_v
cvpred = matrix(NA,nrow=7542,ncol=10) # initialisation de la matrice
# des prédicteurs
for (k in 1:10)
  for (v in 1:6)
  {
    sample1 = Tdata.train[which(fold!=v),2:9]
    sample2 = Tdata.train[which(fold==v),2:9]
    class1 = Tdata.train[which(fold!=v),1]
    cvpred[which(fold==v),k] = knn(sample1,sample2,class1,k=k)
  }
class = as.numeric(Tdata.train[,1])
# display misclassification rates for k=1:10
z=apply(cvpred,2,function(x) sum(class!=x)) # calcule l'erreur de classif
#normalement 7 le meilleur K
x[i]=which.min(z)
}
i=Mode(x)

#Prédir par l'arbre de décisions
knn.predictions.signal=knn(Tdata.train[,2:9],Tdata.eval[,2:9],Tdata.train[,1],k=i)
t=table(knn.predictions.signal,Tdata.eval[,"signal"])
#Evaluer l'erreur de la prédiction
Erreur=(sum(t)-t[1,1]-t[2,2]-t[3,3])/sum(t)

regr.eval(Tdata.eval[, "signal"], knn.predictions.signal, train.y = Tdata.eval[,"signal"])


#prédiction par l'arbre de décision
library(rpart)
rt.signal = rpart(signal ~ ., data = Tdata.train)
rt.prediction.signal=predict(rt.signal,Tdata.eval,type="class")
t=table(rt.prediction.signal,Tdata.eval[,"signal"])
#erreur de prediction
Erreur=(sum(t)-t[1,1]-t[2,2]-t[3,3])/sum(t)
#L'arbre de décision obtenu
prp(rt.signal,extra=1)


