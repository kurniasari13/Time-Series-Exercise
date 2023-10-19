import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
#import statsmodels.api as sm
from scipy.stats.distributions import chi2
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

### The ACF ###

sgt.plot_acf(df.market_value, zero=False, lags=40)
plt.title("The ACF for Prices FTSE", size=20)
plt.show()
# hasil: the two time series are similar in the way they behave and the way past values effect present one
# secara umum prefer parsimonious model: model yang sederhana dan have fewer total lags (reasonable predictors  and prevent overfitting)

### The PACF ###

sgt.plot_pacf(df.market_value, zero=False, lags=40, alpha=0.05, method=('ols'))
plt.title("The PACF for Prices FTSE", size=20)
plt.show()
# hasil: lag 25 keatas tidak signifikan (koefisien sangat mendekati 0 sehingga dampaknya sangat kecil, so ignore it)
# buat model dengan lags kurang dari 25
# lags diatas 20 adalah negatif, kemungkinan ada cyclical change every month
# (lags diatas 20): notes bussiness day satu bulan ada 22 hari, tp dampak ini overshowdow dgn dampak dari lags yang dekat
# lags 1 dampkanya yang paling besar

### The AR(1) Model ###

model_ar = AutoReg(df.market_value, lags=1) #1 = number of lags, 0=not taking any of the residual values into consideration
# simple AR(1) model: order(1,0) : Xt= C + koeff * Xt-1 + resid
results_ar = model_ar.fit()
print(results_ar.summary())

###### Try fitting a more complex model for greater accuracy 
model_ar_2 = AutoReg(df.market_value, lags=2)
results_ar_2 = model_ar_2.fit()
print(results_ar_2.summary())

model_ar_3 = AutoReg(df.market_value, lags=3)
results_ar_3 = model_ar_3.fit()
print(results_ar_3.summary())

model_ar_4 = AutoReg(df.market_value, lags=4)
results_ar_4 = model_ar_4.fit()
print(results_ar_4.summary())

### LLR Test ###

def LLR_test(mod_1, mod_2, DF=1):
    L1= mod_1.fit().llf
    L2= mod_2.fit().llf
    LR= (2*(L2-L1))
    p= chi2.sf(LR,DF).round(3)
    return p
print(LLR_test(model_ar, model_ar_2))
print(LLR_test(model_ar_2, model_ar_3))
print(LLR_test(model_ar_3, model_ar_4))

model_ar_4 = AutoReg(df.market_value, lags=4)
results_ar_4 = model_ar_4.fit()
print(results_ar_4.summary())
print("LLR Test:"+ str(LLR_test(model_ar_3, model_ar_4)))

model_ar_5 = AutoReg(df.market_value, lags=5)
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("LLR Test:"+ str(LLR_test(model_ar_4, model_ar_5)))

model_ar_6 = AutoReg(df.market_value, lags=6)
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("LLR Test:"+ str(LLR_test(model_ar_5, model_ar_6)))

model_ar_7 = AutoReg(df.market_value, lags=7)
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary())
print("LLR Test:"+ str(LLR_test(model_ar_6, model_ar_7)))

model_ar_8 = AutoReg(df.market_value, lags=8)
results_ar_8 = model_ar_8.fit()
print(results_ar_8.summary())
print("LLR Test:"+ str(LLR_test(model_ar_7, model_ar_8)))

model_ar_9 = AutoReg(df.market_value, lags=9)
results_ar_9 = model_ar_9.fit()
print(results_ar_9.summary())
print("LLR Test:"+ str(LLR_test(model_ar_8, model_ar_9)))

model_ar_10 = AutoReg(df.market_value, lags=10)
results_ar_10 = model_ar_10.fit()
print(results_ar_10.summary())
print("LLR Test:"+ str(LLR_test(model_ar_9, model_ar_10)))

model_ar_11 = AutoReg(df.market_value, lags=11)
results_ar_11 = model_ar_11.fit()
print(results_ar_11.summary())
print("LLR Test:"+ str(LLR_test(model_ar_10, model_ar_11)))

## komples model berhenti jika LLR test tidak signifikan dan koefisien untuk lags terakhir juga tidak signifikan
## model dengan lags 8 tidak LLR dan koefisien tidak signifikan, sehingga kita berhenti dan menggunakan model dengan lags 7

#Cek LLR lagi untuk model lags 7
print("LLR Test:" + str(LLR_test(model_ar, model_ar_7, DF=6)))

## kesimpulan AR1 > AR 2, but more lags (AR7) > AR 1, sering terjadi saat menggunakan AR model untuk predict non stationary data