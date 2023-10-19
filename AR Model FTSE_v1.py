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

model_ar = ARIMA(df.market_value, order=(1,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
# simple AR(1) model: order(1,0) : Xt= C + koeff * Xt-1 + resid
results_ar = model_ar.fit()
print(results_ar.summary())

###### Try fitting a more complex model for greater accuracy 
model_ar_2 = ARIMA(df.market_value, order=(2,0,0))
results_ar_2 = model_ar_2.fit()
print(results_ar_2.summary())

model_ar_3 = ARIMA(df.market_value, order=(3,0,0))
results_ar_3 = model_ar_3.fit()
print(results_ar_3.summary())

model_ar_4 = ARIMA(df.market_value, order=(4,0,0))
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

#model_ar_4 = ARIMA(df.market_value, order=(4,0,0))
#results_ar_4 = model_ar_4.fit()
#print(results_ar_4.summary())
#print("LLR Test:"+ str(LLR_test(model_ar_3, model_ar_4)))

model_ar_5 = ARIMA(df.market_value, order=(5,0,0))
results_ar_5 = model_ar_5.fit()
print(results_ar_5.summary())
print("LLR Test:"+ str(LLR_test(model_ar_4, model_ar_5)))

model_ar_6 = ARIMA(df.market_value, order=(6,0,0))
results_ar_6 = model_ar_6.fit()
print(results_ar_6.summary())
print("LLR Test:"+ str(LLR_test(model_ar_5, model_ar_6)))

model_ar_7 = ARIMA(df.market_value, order=(7,0,0))
results_ar_7 = model_ar_7.fit()
print(results_ar_7.summary())
print("LLR Test:"+ str(LLR_test(model_ar_6, model_ar_7)))

model_ar_8 = ARIMA(df.market_value, order=(8,0,0))
results_ar_8 = model_ar_8.fit()
print(results_ar_8.summary())
print("LLR Test:"+ str(LLR_test(model_ar_7, model_ar_8)))

#model_ar_9 = ARIMA(df.market_value, order=(9,0,0))
#results_ar_9 = model_ar_9.fit()
##print(results_ar_9.summary())
#print("LLR Test:"+ str(LLR_test(model_ar_8, model_ar_9)))

#model_ar_10 = ARIMA(df.market_value, order=(10,0,0))
#results_ar_10 = model_ar_10.fit()
#print(results_ar_10.summary())
#print("LLR Test:"+ str(LLR_test(model_ar_9, model_ar_10)))

#model_ar_11 = ARIMA(df.market_value, order=(11,0,0))
#results_ar_11 = model_ar_11.fit()
#print(results_ar_11.summary())
#print("LLR Test:"+ str(LLR_test(model_ar_10, model_ar_11)))

## komples model berhenti jika LLR test tidak signifikan dan koefisien untuk lags terakhir juga tidak signifikan
## model dengan lags 8 tidak LLR dan koefisien tidak signifikan, sehingga kita berhenti dan menggunakan model dengan lags 7

#Cek LLR lagi untuk model lags 7 dengan model awal 
print("LLR Test:" + str(LLR_test(model_ar, model_ar_7, DF=6)))

## kesimpulan AR1 > AR 2, but more lags (AR7) > AR 1, sering terjadi saat menggunakan AR model untuk predict non stationary data

### Analisis Residual of Prices ###
df['res_price'] = results_ar_7.resid
print(df.res_price.mean())
## hasil: mean= 0.35224, mean mendekati 0 yang artinya secara rata-rata model AR(7) performs well
print(df.res_price.var())
## hasil: variance 4005.9446, high variance menandakan bahwa residual tidak terkonsentrasi di sekitar the mean of zero, but are all over the place
## variance yang tinggi disebabkan model perform poorly saat memprediksi non stationary data

### DF Test untu stationary residual data ###
print(sts.adfuller(df.res_price))
## hasil: p value 0 dan statistik lebih kecil dari alpha, artinya residual is stationary, fits with white noise

### ACF Test untuk Residual ###
sgt.plot_acf(df.res_price, zero=False, lags=40)
plt.title("ACF of Residuals for Prices", size=20)
plt.show()
## hasil: most koefisien fall within the blue region, artinya tidak signifikan dan fits with karakter white noise 
## ada beberapa koefisien yg signifikan, artinya ada predictor lain yg lebih baik

### Plotting Residual vs Prices ###
df.market_value.plot(figsize=(20,5))
plt.title("Prices", size=20)
plt.show()

df.res_price[1:].plot(figsize=(20,5))
plt.title("Residual of Prices", size=20)
plt.show()
## hasil: plot residual berbeda dengan plot price, tidak ada pattern yang jelas
## berdasarkan analisis, our choice of model seems correct