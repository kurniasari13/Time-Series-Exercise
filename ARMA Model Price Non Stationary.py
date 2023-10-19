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

def LLR_test(mod_1, mod_2, DF=1):
    L1= mod_1.fit().llf
    L2= mod_2.fit().llf
    LR= (2*(L2-L1))
    p= chi2.sf(LR,DF).round(3)
    return p

### The ACF ###

sgt.plot_acf(df.market_value, unbiased= True, zero=False, lags=40)
plt.title("The ACF for Prices FTSE", size=20)
plt.show()
## hasil: all koeff signifikan

### The PACF ###

sgt.plot_pacf(df.market_value, zero=False, lags=40, alpha=0.05, method=('ols'))
plt.title("The PACF for Prices FTSE", size=20)
plt.show()
## hasil: 6 dari 7 koeff pertama signifikan

### ARMA(1) Model for Returns ###
model_ret_ar_1_ma_1 = ARIMA(df.market_value, order=(1,0,1))
results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()
print(results_ret_ar_1_ma_1.summary())
# hasil di video: p value koeff ar dan konstanta are significant, koeef ma tidak signifikan
# simple model is not best fit

df['res_ar_1_ma_1'] = results_ret_ar_1_ma_1.resid

sgt.plot_acf(df.res_ar_1_ma_1, zero=False, lags=40)
plt.title("ACF of Residuals of Prices", size=20)
plt.show()
## hasil: 5 dari 6 koeff pertama signifikan, we must amend this, accounting for up to that many lags in our model

### Fitting Higher Lags for ARMA Model ###
model_ar_6_ma_6 = ARIMA(df.market_value, order=(6,0,6))
results_ar_6_ma_6 = model_ar_6_ma_6.fit()
print(results_ar_6_ma_6.summary())
## hasil: ada error: the initial AR Koeff are not stationary
## ada beberapa solusi: 1. inducing stationary dengan transforming the data into returns
## 2. choosing a different order, 3. setting the initial parameters 
## cara ketiga dengan menambahkan argumen di dalam fit model, contoh fit(start_ar_lags= 7) , tidak harus angka 7, bebas angka berapa saja asalkan greater that the AR order/lags
## if the method fails to compile, we continue increasing the starting lags until we get our summarized table
## dalam kasus ini bisa mencoba starting lags antaara 7 hingga 11 until the method ran without any errors

## hasil: koeff banyak yang tidak signifikan, we should lower the total number of lags
## we should try all models which contain either 6AR lags or 6 MA lags
## untuk mempercepat, lecturer memberitahu bahwa ada 2 model yang koeef semua sgnifikan yaitu ARMA(5,6) dan ARMA(6,1)
## note ARMA(4,6) contains only a single non significant coeffici ent

model_ar_5_ma_6 = ARIMA(df.market_value, order=(5,0,6))
results_ar_5_ma_6 = model_ar_5_ma_6.fit()
print(results_ar_5_ma_6.summary())

model_ar_6_ma_1 = ARIMA(df.market_value, order=(6,0,1))
results_ar_6_ma_1 = model_ar_6_ma_1.fit()
print(results_ar_6_ma_1.summary())

print("\n ARMA(5,6): \t LL = ", results_ar_5_ma_6.llf, "\t AIC = ", results_ar_5_ma_6.aic)
print("\n ARMA(6,1): \t LL = ", results_ar_6_ma_1.llf, "\t AIC = ", results_ar_6_ma_1.aic)
## hasil di video: ARMA(5,6) have higher log likelihood dan lower AIC than ARMA(6,1)

df['res_ar_5_ma_6'] = results_ar_5_ma_6.resid

sgt.plot_acf(df.res_ar_5_ma_6, zero=False, lags=40)
plt.title("ACF of Residuals of Prices", size=20)
plt.show()
## hasil di video: hanya 3 koeff yang signifikan, some negatif and some positif, seems to be no apparent pattern whatsoever
## in short the residuals resemble white noise

## ada pertnyataan bahwa ARMA model tidak bagus untuk data non stationary, coba bandingkan hasil dari ARMA price dengan ARMA returns
## log likelihood lebih besar dan AIC lebih kecil jika datanya stationary
# even though we can model price using ARMA model, they perform much worse compared to their ability to estimate stationary data
