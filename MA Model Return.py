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
#periode pertama tidak memiliki retunrs karena tidak memiliki data from previous period, so we need drop it

#### DF Test ###
print(sts.adfuller(df.returns[1:]))
##hasil: z statistik lebih kecil dari alpha (reject Ho), artinya data is stationary 

### The ACF ###
sgt.plot_acf(df.returns[1:], zero=False, lags=40)
plt.title("The ACF for Prices FTSE Returns", size=20)
plt.show()
# hasil: variasinya besar, ada positif dan negatif, ada yang signifikan dan ada yang tidak
# lags 6: consecutive values move in differet direction, artinya return over the entire week relevant with current value
# nagatif acf: some form of natural adjustment occuring n the market

### MA(1) Model for Returns ###
model_ret_ma_1 = ARIMA(df.returns[1:], order=(0,0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
print(results_ret_ma_1.summary())

## hasil di video: koeff lag 1 residual tidak signifikan, hal ini tidak mengejutkan karena lags 1 di ACF plot tidak signifikan juga

### Fitting Higher Lags for MA Model ###
model_ret_ma_2 = ARIMA(df.returns[1:], order=(0,0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
print(results_ret_ma_2.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_1 , model_ret_ma_2)))
## hasil: lags 1 dan lags 2 signifikan

model_ret_ma_3 = ARIMA(df.returns[1:], order=(0,0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_2 , model_ret_ma_3)))

model_ret_ma_4 = ARIMA(df.returns[1:], order=(0,0,4))
results_ret_ma_4 = model_ret_ma_4.fit()
print(results_ret_ma_4.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_3 , model_ret_ma_4)))

model_ret_ma_5 = ARIMA(df.returns[1:], order=(0,0,5))
results_ret_ma_5 = model_ret_ma_5.fit()
print(results_ret_ma_5.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_4 , model_ret_ma_5)))

model_ret_ma_6 = ARIMA(df.returns[1:], order=(0,0,6))
results_ret_ma_6 = model_ret_ma_6.fit()
print(results_ret_ma_6.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_5 , model_ret_ma_6)))

model_ret_ma_7 = ARIMA(df.returns[1:], order=(0,0,7))
results_ret_ma_7 = model_ret_ma_7.fit()
print(results_ret_ma_7.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_6 , model_ret_ma_7)))

## hasil: koeff lag 7 dan LLR test tidak signifikan, based on rule of thumb, seharusnya model selection berhenti di lags 6,
## tetapi di ACF, lags 8 memiliki koefisien yang signifikan, sehingga kita harus menganalisis lags 8 juga

model_ret_ma_8 = ARIMA(df.returns[1:], order=(0,0,8))
results_ret_ma_8 = model_ret_ma_8.fit()
print(results_ret_ma_8.summary())
print("LLR test p-value = " + str(LLR_test(model_ret_ma_7 , model_ret_ma_8)))

# hasil: MA6 > MA7, MA8 > MA7, but MA8 > MA7 ??? test using LLR Test
print(LLR_test(model_ret_ma_6, model_ret_ma_8, DF=2))

### Analisis Residual for Returns ###
df['res_ret_ma_8'] = results_ret_ma_8.resid[1:]
print("The mean of the residual is " + str(round(df.res_ret_ma_8.mean(), 3)) + "\nThe Variance of the residuals is " + str(round(df.res_ret_ma_8.var(),3)))
from math import sqrt
print(round(sqrt(df.res_ret_ma_.var()),3)) #standar deviation
# mean -0,0, variance 1,356, standar deviasi 1,164
# Gaussian white noise implies normality (distribusi normal), 68, 95, 99.7 = what part of the data is spread within 1, 2, and 3 standar deviations away from the mean in  either direction
# 3 standar deviation (3,5 = 3 * 1,164) in either direction, artinya we expect that most return residual will be between - 3,5 dan 3,5 (dikali 3 karena 3 standar deviation / rule 99.7)
# worst case scenario: 7 percentage points off when prediction the returns for a market index (3.5 * 2)
# error 7 persen is great when dealing with profit and loss

df.res_ret_ma_8[1:].plot(figsize=(20,5))
plt.title("Residuals of Returns", size=23)
plt.show()
## hasil: residuals rather random

### Cek Stationary of Residuals Returns ###
print(sts.adfuller(df.res_ret_ma_8[1:]))
## hasil: p value 0 artinya residual return is stationary

### Plot ACF, jika tidak siginifikan artinya memenuhi asumsi white noise ###
sgt.plot_acf(df.res_ret_ma_8[2:], zero=False, lags=40)
plt.title("ACF of Residuals for Returns", size=23)
plt.show()
## hasil: most koeff not signifikan, khususnya lags 17 pertama tidak signifikan
## ACF residual untuk lags 8 mendekati zero, hal ini wajar karena lags 8 masuk ke model MA(8) 
## ACF residual: lags antara 9-17 juga insignifikan yang artinya model perform well
## semakin jauh ke masa lalu, nilai dan eror menjadi kurang relevan