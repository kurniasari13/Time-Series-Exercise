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

### The ACF of Prices ###
sgt.plot_acf(df.market_value, zero=False, lags=40)
plt.title("The ACF for Prices FTSE", size=20)
plt.show()
## hasil: koeff dari 40 lags are significant
## telihat bahwa model dengan banyak lags lebih baik daripada mdoel dengan sedikit lags, sehingga model dengan lags infinitive terlihat bagus
## karena lags infinitive tidak ada, terlihat bahwa tidak ada MA Model would be a good estimate of price

### MA Model for Prices ###
model_ma_1 = ARIMA(df.market_value, order=(0,0,1))
results_ma_1 = model_ma_1.fit()
print(results_ma_1.summary())
## hasil: koff semua siginifikan, koeff lag 1 hampir mendekati 1 (0,9573): our model tries to keep almost the entire magnitude of the error term from the last period
## koeff lags 1 hampir mendekati 1: artinya setiap model akan memprediksi value, it actually tries to maximize on the error from the last time
## since this is a simple model with 1 lags, the error term contains all the information from the other lags

### kesimpulan MA Model do not perform well for not stationary data
### MA Model perform well dengan stationary data jika Ma model juga menggunakan previoues periods's values

model_ma_8 = ARIMA(df.market_value, order=(0,0,8))
results_ma_8 = model_ma_8.fit()
print(results_ma_8.summary())

### Analisis Residual ###
df['res_ma_8'] = results_ma_8.resid[1:]

df.res_ma_8[1:].plot(figsize=(20,5))
plt.title("Residual for Prices", size=23)
plt.show()

sgt.plot_acf(df.res_ma_8[2:], zero=False, lags=40)
plt.title("ACF of Residuals for Prices", size=23)
plt.show()
## hasil: koeff residual mostly signifikan

print(sts.adfuller(df.res_ma_8[1:]))
## hasil: residual bersifat non-stationary (statistik lebih besar daripada alpha or do not recejt Ho and Ho: data is non-stationary)