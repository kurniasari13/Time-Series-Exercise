import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
#import statsmodels.api as sm
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts
import seaborn as sns 
sns.set()

### Preprocessing Time Series ###
raw_csv_data = pd.read_csv('D:/DATA ANALYST/belajar_python/TIME SERIES/Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)
df_comp.set_index("date", inplace=True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method='ffill')

df_comp['market_value'] = df_comp.ftse
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### LLR Test ###
def LLR_test(mod_1, mod_2, DF=1):
    L1= mod_1.fit().llf
    L2= mod_2.fit().llf
    LR= (2*(L2-L1))
    p= chi2.sf(LR,DF).round(3)
    return p

### DF Test untuk cek stationary data prices ###
print(sts.adfuller(df.market_value))
##hasil: z statistik lebih besar dari p value (do not reject Ho), artinya data is non-stationary
##solusi: ubah data dari price become returns, so it will fit stationary assumes

### Using Returns Instead Prices ###
df['returns'] = df.market_value.pct_change(1).mul(100) 
print(df.head())
#pct_change: mencari returns dan mul: perkalian
#periode pertama tidak memiliki returns karena tidak memiliki data from previous period, so we need drop it

#### DF Test ###
print(sts.adfuller(df.returns[1:]))
##hasil: z statistik lebih kecil dari alpha (reject Ho), artinya data is stationary 

### The ACF ###
sgt.plot_acf(df.returns[1:], zero=False, lags=40)
plt.title("The ACF for Prices FTSE Returns", size=20)
plt.show()
# hasil: variasinya besar, ada positif dan negatif, ada yang signifikan dan ada yang tidak
# lags 6: consecutive values move in differet direction, artinya return over the entire week relevant with current value
# nagatif acf: some form of natural adjustment occuring in the market

### The PACF ###

sgt.plot_pacf(df.returns[1:], zero=False, lags=40, alpha=0.05, method=('ols'))
plt.title("The PACF for Returns", size=20)
plt.show()

## hasil: ARMA model will contain no more than six AR lags and at most eight MA lags ( if using them simultaneously would be redundant)

### ARMA(1) Model for Returns ###
model_ret_ar_1_ma_1 = ARIMA(df.returns[1:], order=(1,0,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
print(results_ret_ar_1_ma_1.summary())

## hasil: hanya konstanta yang p valuenya tidak signifikan
## koeff AR lags positif dan mendekati 1: returns move in trends of consecutive positive or negative values
## AR lags: jika diartikan ke price, positive menandakan periods of persistent increase, meanwhile negative menandakan period of persistent decrease 
## MA lags negatif: sulit diinterpretasikan, we should be moving away from the past period values rather than trying to use them as targets for calibration

### LLR Test ###
model_ret_ar_1 = ARIMA(df.returns[1:], order=(1,0,0))
model_ret_ma_1 = ARIMA(df.returns[1:], order=(0,0,1))

print("ARMA vs AR", LLR_test(model_ret_ar_1 , model_ret_ar_1_ma_1))
print("ARMA vs MA", LLR_test(model_ret_ma_1 , model_ret_ar_1_ma_1))
## hasil: model ARMA lebih baik daripada model AR dan MA
## using past error in conjunction with past values result in much better estimators
## other interpretion: our past estimator (past value and past error) has performed better as predictors than actual past values


### Fitting Higher Lags for ARMA Model ###
model_ret_ar_8_ma_6 = ARIMA(df.returns[1:], order=(8,0,6))
results_ret_ar_8_ma_6 = model_ret_ar_8_ma_6.fit()
print(results_ret_ar_8_ma_6.summary())
## hasil: 10 dari 14 koeff tidak signifikan, this os proof that the model is unnecessarily complicated
## if we assume ARMA(4,3) is half the ARMA(8,6), starting with a simpler ARMA(3,3) sound all the more sensible

model_ret_ar_3_ma_3 = ARIMA(df.returns[1:], order=(3,0,3))
results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()
print(results_ret_ar_3_ma_3.summary())
print(LLR_test(model_ret_ar_1_ma_1, model_ret_ar_3_ma_3, DF=4)) #LLR test 0,00
## default DF=1
## hasil di video: lags 1 MA tidak signifikan, this suggests that it may be irrelevant, so a lower lag model might be of better use
## hasil LLR test suggest ARMA(3,3) make better estimations that ARMA(1,1)
## MA koeff ada yang negatif
## If ARMA(3,3) is not the optimal model, then we expect the best fit to be somewhere between ARMA(1,1) and ARMA(3,3) 

model_ret_ar_3_ma_2 = ARIMA(df.returns[1:], order=(3,0,2))
results_ret_ar_3_ma_2 = model_ret_ar_3_ma_2.fit()
print(results_ret_ar_3_ma_2.summary())
print(LLR_test(model_ret_ar_3_ma_2, model_ret_ar_3_ma_3, DF=1))
## hasil LLR test: 0,035, better ARMA(3,3), but we must try other model

## hasil: all koeff kecuali konstanta are significant
## as the lags increase the absolute values or koeff for AR and MA decrease, this is support the idea that the further back in time we go, the less relevant values and error become
## this does not necessary hold true for every model, but such a trend makes our predictions seem much more realistic
## MA lags koeff positive, this is suggest calibration effort, contoh if prediction lower thant actual values, maka error akan positif, positive error dikali positive koeff, we get MA component with a plus sign
## this results in an increase in the value of our predictions for the next period, so we try to close the gap to the actual value, hoping the pattern will translate into the future
## if MA lags koeff negatif: it means our predictions are higher than the actual values for that period, negative error dikali positif koeff leave us with a negatve MA component, so the next prediction would be lower 
## so we are once again trying to get closer to the actual values

## AR koeff negatif: match our expectation of an efficient market.
## if returns for a given period are positive, then multiplying the values by a negative koeff/factor will move their effect in the opposite direction
## we exoect an efficient market to have a mean of zero over time, therefore every period of positive returns is followed by one with nagative returns
## the assumption is that it allows us to remain close to the mean of zero, regardless of the starting and ending periods in our sample
## this is yet another case which shows why ARMA models are good estimates of stationary data

model_ret_ar_2_ma_3 = ARIMA(df.returns[1:], order=(2,0,3))
results_ret_ar_2_ma_3 = model_ret_ar_2_ma_3.fit()
print(results_ret_ar_2_ma_3.summary())
print(LLR_test(model_ret_ar_2_ma_3, model_ret_ar_3_ma_3, DF=1))
## hasil di video: sama seperti ARMA(3,3), MA lags ada yang tidak signifikan, so must avoid this model
## LLR Test 0,042

model_ret_ar_3_ma_1 = ARIMA(df.returns[1:], order=(3,0,1))
results_ret_ar_3_ma_1 = model_ret_ar_3_ma_1.fit()
print(results_ret_ar_3_ma_1.summary())
print(LLR_test(model_ret_ar_3_ma_1, model_ret_ar_3_ma_2, DF=1)) ## 0,01
## hasil: koeff semua signifikan kecuali konstanta, koef ar negatifm koeff dan koeff ma positif, sama dengan ARMA(3,2)
## ARMA(3,2) > ARMA(3,1)

model_ret_ar_1_ma_3 = ARIMA(df.returns[1:], order=(1,0,3))
results_ret_ar_1_ma_3 = model_ret_ar_1_ma_3.fit()
print(results_ret_ar_1_ma_3.summary())
print(LLR_test(model_ret_ar_1_ma_3, model_ret_ar_3_ma_2, DF=1))
## LLR tidak bisa dilakukan karena lags ma 3 lebih besar daripada lags ma 2
##we can manually compare the log likelihoodand AICs of both model
print("\n ARMA(3,2): \tLL = ", results_ret_ar_3_ma_2.llf, "\tAIC = ", results_ret_ar_3_ma_2.aic)
print("\n ARMA(1,3): \tLL = ", results_ret_ar_1_ma_3.llf, "\tAIC = ", results_ret_ar_1_ma_3.aic)
## hasil ARMA(3,2) > ARMA(1,3)
## semua koeff signifikan

model_ret_ar_2_ma_2 = ARIMA(df.returns[1:], order=(2,0,2))
results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()
print(results_ret_ar_2_ma_2.summary())
print(LLR_test(model_ret_ar_2_ma_2, model_ret_ar_3_ma_2, DF=1)) ## 0,00
## koeff ar dan ma ada yang tidak signifikan, avoid this model
## koeff ar ada yg positif dan koeff ma ada yang negatif

model_ret_ar_2_ma_1 = ARIMA(df.returns[1:], order=(2,0,1))
results_ret_ar_2_ma_1 = model_ret_ar_2_ma_1.fit()
print(results_ret_ar_2_ma_1.summary())
print(LLR_test(model_ret_ar_2_ma_1, model_ret_ar_3_ma_2, DF=2)) ## 0,00

model_ret_ar_1_ma_2 = ARIMA(df.returns[1:], order=(1,0,2))
results_ret_ar_1_ma_2 = model_ret_ar_1_ma_2.fit()
print(results_ret_ar_1_ma_2.summary())
print(LLR_test(model_ret_ar_1_ma_2, model_ret_ar_3_ma_2, DF=2)) ## 0,00

##### the best model seem ARMA(3,2), why:
## all significant koeffisiens, outpredicts all less complex alternatives model
## agar lebih yakin, maka perlu analisis residual

### Analisis Residual for Returns ###
df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]

df.res_ret_ar_3_ma_2.plot(figsize=(20,5))
plt.title("Residuals of Returns", size=24)
plt.show()
## hasil: plot residual return ARMA sama dengan plot residual from AR dan MA model
## thus suggest that the volatility in return might not be fully comprehensible if we use only ARMA Model

sgt.plot_acf(df.res_ret_ar_3_ma_2[2:], zero=False, lags=40)
plt.title("ACF of residuals for returns", size=23)
plt.show()
## hasil: we have more significant lags than AR and MA model, lags 5 siginifikan, di AR dan Ma model lags 5 is zero
## therefore, accounting for either returns or residual five periods ago could improve our predictions
## we should start woth ARMA(5,5) = ARMA(5,Q) or ARMA(P,5)

### Reevaluating Model Selection ###
model_ret_ar_5_ma_5 = ARIMA(df.returns[1:], order=(5,0,5))
results_ret_ar_5_ma_5 = model_ret_ar_5_ma_5.fit()
print(results_ret_ar_5_ma_5.summary())
## hasil di video: hanya 3 koeff yang signifikan, taking both returns and residuals 5 periods ago is redundant
## hasil sendiri: 5 koeff signifikan dan 5 koeff tidak signifikan
## we should only focus on one of the two: ARMA(1,5), (2,5), (3,5), (4,5), (5,1), (5,2), (5,3), (5,4)

model_ret_ar_1_ma_5 = ARIMA(df.returns[1:], order=(1,0,5))
results_ret_ar_1_ma_5 = model_ret_ar_1_ma_5.fit()
print(results_ret_ar_1_ma_5.summary())
## hasil: koeff semua signifikan, tetapi koeff ar ada yg positif dan ma ada yang negatif

model_ret_ar_2_ma_5 = ARIMA(df.returns[1:], order=(2,0,5))
results_ret_ar_2_ma_5 = model_ret_ar_2_ma_5.fit()
print(results_ret_ar_2_ma_5.summary())
## hasil: koeff ada yang tidak signifikan

model_ret_ar_3_ma_5 = ARIMA(df.returns[1:], order=(3,0,5))
results_ret_ar_3_ma_5 = model_ret_ar_3_ma_5.fit()
print(results_ret_ar_3_ma_5.summary())
## hasil: koeff ada yang tidak signifikan

model_ret_ar_4_ma_5 = ARIMA(df.returns[1:], order=(4,0,5))
results_ret_ar_4_ma_5 = model_ret_ar_4_ma_5.fit()
print(results_ret_ar_4_ma_5.summary())
## hasil: koeff ada yang tidak signifikan

model_ret_ar_5_ma_1 = ARIMA(df.returns[1:], order=(5,0,1))
results_ret_ar_5_ma_1 = model_ret_ar_5_ma_1.fit()
print(results_ret_ar_5_ma_1.summary())
## hasil: koeff semua signifikan, tetapi koeff ar ada yg positif dan ma ada yang negatif

model_ret_ar_5_ma_2 = ARIMA(df.returns[1:], order=(5,0,2))
results_ret_ar_5_ma_2 = model_ret_ar_5_ma_2.fit()
print(results_ret_ar_5_ma_2.summary())
## hasil: koeff semua signifikan, tetapi koeff ar ada yg positif dan ma ada yang negatif

model_ret_ar_5_ma_3 = ARIMA(df.returns[1:], order=(5,0,3))
results_ret_ar_5_ma_3 = model_ret_ar_5_ma_3.fit()
print(results_ret_ar_5_ma_3.summary())
## hasil: koeff ada yang tidak signifikan

model_ret_ar_5_ma_4 = ARIMA(df.returns[1:], order=(5,0,4))
results_ret_ar_5_ma_4 = model_ret_ar_5_ma_4.fit()
print(results_ret_ar_5_ma_4.summary())
## hasil: koeff ada yang tidak signifikan

## hadil di video: koeff yang semua signifikan hanya model ARMA(1,5) dan ARMA(5,1)
## can not apply LLR test, so we must compare their log likelihood dan AIC values

print("\n ARMA(5,1): \tLL = ", results_ret_ar_5_ma_1.llf, "\tAIC = ", results_ret_ar_5_ma_1.aic)
print("\n ARMA(5,2): \tLL = ", results_ret_ar_5_ma_2.llf, "\tAIC = ", results_ret_ar_5_ma_2.aic)
print("\n ARMA(1,5): \tLL = ", results_ret_ar_1_ma_5.llf, "\tAIC = ", results_ret_ar_1_ma_5.aic)
## hasil di video: ARMA(5,1) have higher log likelihood and lower AIC, mari bandingkan dengan model best lainnya yaitu ARMA(3,2)
## hasil sendiri: ARMA(5,2) have higger log likelihood dan lower AIC
print("\n ARMA(3,2): \tLL = ", results_ret_ar_3_ma_2.llf, "\tAIC = ", results_ret_ar_3_ma_2.aic)
## hasil di video: ARMA(5,1) have higher log likelihood and lower AIC
## kesimpulan di video ARMA(5,1) is the bset model so far\
## kesimpulan sendiri ARMA(5,2) is the best so far

### Analisis Residual for the New Best Model ###
df['res_ret_ar_5_ma_1'] = results_ret_ar_5_ma_1.resid[1:]

df.res_ret_ar_5_ma_1.plot(figsize=(20,5))
plt.title("Residuals of Returns", size=24)
plt.show()

sgt.plot_acf(df.res_ret_ar_5_ma_1[2:], zero=False, lags=40)
plt.title("ACF of residuals for returns", size=23)
plt.show()
## hasil: lags antara 1-17 lags tidak signifikan
## if we want our model to resemble the data set mode closely, we could include even more lags into our model, however this will predispose the model to failure when facing unfamiliar data
## more lags will help us know with confidance how this specific data set move, rather than understand how the actual market returns fluctuate
## the effect of returns and errors tend to diminish over time, the further back in time we go, the less relevant the values are in predicting the future
## since more that 10 of the first lags are not significant, we can see the residuals are pretty much random, which is what we were trying to achieve
