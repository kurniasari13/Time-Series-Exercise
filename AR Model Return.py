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
df = df.iloc[1:]
print(df.head())

#### DF Test ###
print(sts.adfuller(df.returns))
##hasil: z statistik lebih kecil dari alpha (reject Ho), artinya data is stationary 


### The ACF ###

sgt.plot_acf(df.returns, zero=False, lags=40)
plt.title("The ACF for Prices FTSE Returns", size=20)
plt.show()
# hasil: variasinya besar, ada positif dan negatif, ada yang signifikan dan ada yang tidak
# lags 6: consecutive values move in differet direction, artinya return over the entire week relevant with current value
# nagatif acf: some form of natural adjustment occuring n the market


### The PACF ###

sgt.plot_pacf(df.returns, zero=False, lags=40, alpha=0.05, method=('ols'))
plt.title("The PACF for Prices FTSE Returns", size=20)
plt.show()

# pacf similar dengan acf
# prices today often move in the opposite direction of prices yesterday
# we tend to get prices increases following price decreases, which fall in line with our expectation of cyclical cahnges
# semakin banyak lags semakin tidak signifikan yang artinya valuesnya tidak relevant, hal ini terjadi karena dampak mereka sudah di accounted oleh lags terdekat

### The AR(1) Model for Returns ###

ret_model_ar_1 = ARIMA(df.returns, order=(1,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
# simple AR(1) model: order(1,0,1) : Xt= C + koeff * Xt-1 + resid
results_ret_ar_1 = ret_model_ar_1.fit()
print(results_ret_ar_1.summary())
###hasil divideo tidak signifikan
## hasil: the more easyly yesterday's price is effected by higher lags, the more inaccurate its coefficient becomes
## hasil: penyebabnya ada akumulasi penggabungan effect of lagged coefficients
## hasil: harga sekarang tidak hanya dipengaruhi lags 1, tapi juga dipengaruhi higher lags, sehingga jika hanya lags 1 di dalam model maka model kurang kuat dalam memprediksi
### hasil sendiri: lags 1 signifikan yang artinya return hari ini dipengaruhi return kemarin

### Higher Lags AR Models for Returns ###
ret_model_ar_2 = ARIMA(df.returns, order=(2,0,0))
results_ret_ar_2 = ret_model_ar_2.fit()
print(results_ret_ar_2.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_1, ret_model_ar_2)))

ret_model_ar_3 = ARIMA(df.returns, order=(3,0,0))
results_ret_ar_3 = ret_model_ar_3.fit()
print(results_ret_ar_3.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_2, ret_model_ar_3)))

ret_model_ar_4 = ARIMA(df.returns, order=(4,0,0))
results_ret_ar_4 = ret_model_ar_4.fit()
print(results_ret_ar_4.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_3, ret_model_ar_4)))

ret_model_ar_5 = ARIMA(df.returns, order=(5,0,0))
results_ret_ar_5 = ret_model_ar_5.fit()
print(results_ret_ar_5.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_4, ret_model_ar_5)))

ret_model_ar_6 = ARIMA(df.returns, order=(6,0,0))
results_ret_ar_6 = ret_model_ar_6.fit()
print(results_ret_ar_6.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_5, ret_model_ar_6)))

ret_model_ar_7 = ARIMA(df.returns, order=(7,0,0))
results_ret_ar_7 = ret_model_ar_7.fit()
print(results_ret_ar_7.summary())
print("LLR Test:"+ str(LLR_test(ret_model_ar_6, ret_model_ar_7)))

##hasil: model yang terbaik adalah model degan lags 6, model dengan lags 7 LLR Test sudah tidak signifikan dan lagsnya juga tidak signifikan

### Analisis Residual of Returns ###
df['res_ret'] = results_ret_ar_6.resid
print(df.res_ret.mean())
## mean -5,375, mean mendekati 0, artinya in average model is perform well
print(df.res_ret.var())
## variance: 1,355, variance sekitar 1, variance rendah artinya residual terkonsentraidi di area tertentu
##variance yang rendah disebabkan model perform well saat memprediksi stationary data, pilihan model kita is pretty good

### DF Test untu stationary residual data ###
print(sts.adfuller(df.res_ret))
# statistk kurang dari alpha dengan p value 0, artinya residual is stationary,fits with white noise

### ACF Test untuk Residual ###
sgt.plot_acf(df.res_ret, zero=False, lags=40)
plt.title("ACF of Residuals for Prices Returns", size=20)
plt.show()
## hasil: most koefisien fall within the blue region, artinya tidak signifikan dan fits with karakter white noise
## ada beberapa koefisien yg signifikan, artinya ada predictor lain yg lebih baik

### Plotting Residual vs Prices ###
df.returns.plot(figsize=(20,5))
plt.title("Returns of Prices", size=20)
plt.show()

df.res_ret.plot(figsize=(20,5))
plt.title("Residual of Prices Returns", size=20)
plt.show()
## plotnya hampir sama dengan residual of prices, resdiual mostly low kecuali for around the end of 2008 karena adanya resesi (unpredictable event)