import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
#import statsmodels.api as sm
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts
from math import sqrt
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
#del df_comp['spx']
#del df_comp['dax']
#del df_comp['ftse']
#del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### LLR Test ###
def LLR_test(mod_1, mod_2, DF=1):
    L1= mod_1.fit().llf
    L2= mod_2.fit().llf
    LR= (2*(L2-L1))
    p= chi2.sf(LR,DF).round(3)
    return p

### MODEL ARIMA(1,1,1) ###
model_ar_1_i_1_ma_1 = ARIMA(df.market_value, order=(1,1,1))
results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
print(results_ar_1_i_1_ma_1.summary())
## hasil di video: konstanta tidak signifikan sama dnegan hasil summary return, return lebih dekat dengan integration dibandingkan prices

### Analisis Residual ###
df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid

sgt.plot_acf(df.res_ar_1_i_1_ma_1[1:], zero=False, lags=40)
plt.title("ACF of Residuals for ARIMA(1,1,1)", size=20)
plt.show()
## hasil di video: the ACF failed tocomputedue to the missing value first element
## hasil: lags 3 dan 4 signifikan shingga perlu dimasukkan ke model
## buat model dari ARIMA(1,1,1) sampe ke ARIMA(4,1,4)
## untuk mempercepat waktu, hanya ada 5 model yg memenuhi kriteria signifikan yaitu: ARIMA(1,1,2), ARIMA(1,1,3), ARIMA(2,1,1), ARIMA(3,1,1), ARIMA(3,1,2)

model_ar_1_i_1_ma_2 = ARIMA(df.market_value, order=(1,1,2))
results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()
print(results_ar_1_i_1_ma_2.summary())

model_ar_1_i_1_ma_3 = ARIMA(df.market_value, order=(1,1,3))
results_ar_1_i_1_ma_3 = model_ar_1_i_1_ma_3.fit()
print(results_ar_1_i_1_ma_3.summary())

model_ar_2_i_1_ma_1 = ARIMA(df.market_value, order=(2,1,1))
results_ar_2_i_1_ma_1 = model_ar_2_i_1_ma_1.fit()
print(results_ar_2_i_1_ma_1.summary())

model_ar_3_i_1_ma_1 = ARIMA(df.market_value, order=(3,1,1))
results_ar_3_i_1_ma_1 = model_ar_3_i_1_ma_1.fit()
print(results_ar_3_i_1_ma_1.summary())

model_ar_3_i_1_ma_2 = ARIMA(df.market_value, order=(3,1,2))
results_ar_3_i_1_ma_2 = model_ar_3_i_1_ma_2.fit()
print(results_ar_3_i_1_ma_2.summary())

print("\n ARIMA(1,1,1): \t LL = ", results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)
print("\n ARIMA(1,1,2): \t LL = ", results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)
print("\n ARIMA(1,1,3): \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("\n ARIMA(2,1,1): \t LL = ", results_ar_2_i_1_ma_1.llf, "\t AIC = ", results_ar_2_i_1_ma_1.aic)
print("\n ARIMA(3,1,1): \t LL = ", results_ar_3_i_1_ma_1.llf, "\t AIC = ", results_ar_3_i_1_ma_1.aic)
print("\n ARIMA(3,1,2): \t LL = ", results_ar_3_i_1_ma_2.llf, "\t AIC = ", results_ar_3_i_1_ma_2.aic)
## hasil: ARIMA(1,1,3) menjadi model dengan log likelihood tertingg dan AIC terkecil
print("\n LLR test p-value = " + str(LLR_test(model_ar_1_i_1_ma_2 , model_ar_1_i_1_ma_3)))
print("\n LLR test p-value = " + str(LLR_test(model_ar_1_i_1_ma_1 , model_ar_1_i_1_ma_3, DF=2)))
## hasil LLR test keduanya signifikan yang artinya komplek model lebih baik dalam mengestimasi daripada model yang simple

### Analisis Residual ###
df['res_ar_1_i_1_ma_3'] = results_ar_1_i_1_ma_3.resid.iloc[:]

sgt.plot_acf(df.res_ar_1_i_1_ma_3[1:], zero=False, lags=40)
plt.title("ACF of Residuals for ARIMA(1,1,3)", size=20)
plt.show()
## hasil: ARIMA(1,1,3) memiliki koeff yang signifikan lebih sedikit dibandingkan ARIMA(1,1,1), however, the sixth one is still highly significant
## therefore, there might exist a better model, which goes up to 6 lags back
## we must check all models from ARIMA(1,1,1) to ARIMA(6,1,6)
## total ada 20 model, hanya 2 model yang yielded only significant values yaitu ARIMA(6,1,3) dan ARIMA(5,1,1)

model_ar_5_i_1_ma_1 = ARIMA(df.market_value, order=(5,1,1))
results_ar_5_i_1_ma_1 = model_ar_5_i_1_ma_1.fit()
print(results_ar_5_i_1_ma_1.summary())

model_ar_6_i_1_ma_3 = ARIMA(df.market_value, order=(6,1,3))
results_ar_6_i_1_ma_3 = model_ar_6_i_1_ma_3.fit()
print(results_ar_6_i_1_ma_3.summary())

print("\n ARIMA(1,1,3): \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)
print("\n ARIMA(5,1,1): \t LL = ", results_ar_5_i_1_ma_1.llf, "\t AIC = ", results_ar_5_i_1_ma_1.aic)
print("\n ARIMA(6,1,3): \t LL = ", results_ar_6_i_1_ma_3.llf, "\t AIC = ", results_ar_6_i_1_ma_3.aic)
## hasil di video: ARIMA(6,1,3) memiliki log likelihood tertinggi dan AIC terkecil
## lakukan tes LLR, jika ada error tambahkan argumen di dalam fityiatu start_ar_lags=11

print("\n LLR test p-value = " + str(LLR_test(model_ar_1_i_1_ma_3 , model_ar_6_i_1_ma_3, DF=5)))
## hasil: LLR test signifikan yaitu p value sebesar 0.017
print("\n LLR test p-value = " + str(LLR_test(model_ar_5_i_1_ma_1 , model_ar_6_i_1_ma_3, DF=3)))
## hasil: LLR test tidak signifikan, p value sebesar 0.113, artinya the higher model fails the test at the 1, 5 and even 10 percent significance level
## this is an indicator that the more complex ARIMA does not yield a significantly higher log likelihood compared to the simpler alternative

### Analisis Residual ###
df['res_ar_5_i_1_ma_1'] = results_ar_5_i_1_ma_1.resid

sgt.plot_acf(df.res_ar_5_i_1_ma_1[1:], zero=False, lags=40)
plt.title("ACF of Residuals for ARIMA(5,1,1)", size=20)
plt.show()
## hasil: 1-15 lags tidak ada yang signifikan, lags 15 keatas ada yang signifikan, tapi abaikan saja karena the further back in time we go, the less relevant the values become
## include all lags, residual bisa menjadi white noise, tapi akan terjadi overfitting
## so far ARIMA(5,1,1) as the best estimator

### Model with Higher Levels of Integration ###
df['delta_prices'] = df.market_value.diff(1)

model_delta_ar_1_i_1_ma_1 = ARIMA(df.delta_prices[1:], order=(1,0,1))
results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
print(results_delta_ar_1_i_1_ma_1.summary())
## hasil: summary model diatas identikal dengan model ARIMA(1,1,1), artinya data yang kita buat secara mnual sama dengan data yang diintegrated otomatis oleh statsmodels

### Augmented Dickey Fuller Test ###
print(sts.adfuller(df.delta_prices[1:]))
## hasil: statistik lebih kecil dari alpha dan p value 0.0, artinya reject Ho yang artinya data is stasionary
## no need for additional layers of integration

### ARIMAX MODEL ###
model_ar_1_i_1_ma_1_Xspx = ARIMA(df.market_value, exog= df.spx ,order=(1,1,1))
results_ar_1_i_1_ma_1_Xspx = model_ar_1_i_1_ma_1_Xspx.fit()
print(results_ar_1_i_1_ma_1_Xspx.summary())
## hasil di video: p value untuk spx tidak signifikan, but maybe it's because we have not included all the indexes
## not all models are used for predicting future prices or returns, many investors often seek stability

### SARIMAX MODEL ###
from statsmodels.tsa.statespace.sarimax import SARIMAX
##SARIMAX(1,0,1)(2,0,1,5) S&p

model_sarimax = SARIMAX(df.market_value, exog= df.spx ,order=(1,0,1), seasonal_order=(2,0,1,5))
results_sarimax = model_sarimax.fit()
print(results_sarimax.summary())