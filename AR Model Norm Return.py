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

### DF Test untuk cek stationary data pricess ###

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

#### model dari berbagai pasar saham di dunia dapat dibandingkan selama semua data di setiap model sudah dinormalize

### NORMALIZE OF PRICES ###
## di TS: normalize dengan cara menghitung percentage harga terhadap harga awal atau harga benchmark
benchmark = df.market_value.iloc[0]
df['norm'] = df.market_value.div(benchmark).mul(100)

### Test DF untuk stationary ###
print(sts.adfuller(df.norm))
## data normalize price ternyata tidak stationary, sehingga kita tidak dapat menggunakan data ini untuk AR Model

### NORMALIZE OF RETURNS ###
bench_ret = df.returns.iloc[0]
df['norm_ret'] = df.returns.div(bench_ret).mul(100)

### Test DF untuk stationary ###
print(sts.adfuller(df.norm_ret))
##hasil: data  normalize return ternyata stationary sama dengan data return (normalize does not effect stationary)

### The AR(1) Model fot Normalize Returns ###
model_norm_ret_ar_1 = ARIMA(df.norm_ret, order=(1,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
results_norm_ret_ar_1 = model_norm_ret_ar_1.fit()
print(results_norm_ret_ar_1.summary())
## hasil: hasilnya sama dengan model return tanpa normalize, yang berbeda hanya nilai konstanta

### Higher Lags AR Models for Normalize Returns ###
model_norm_ret_ar_2 = ARIMA(df.norm_ret, order=(2,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
results_norm_ret_ar_2 = model_norm_ret_ar_2.fit()
print(results_norm_ret_ar_2.summary())

model_norm_ret_ar_6 = ARIMA(df.norm_ret, order=(6,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
results_norm_ret_ar_6 = model_norm_ret_ar_6.fit()
print(results_norm_ret_ar_6.summary())

model_norm_ret_ar_7 = ARIMA(df.norm_ret, order=(7,0,0)) #1 = number of lags, 0=not taking any of the residual values into consideration
results_norm_ret_ar_7 = model_norm_ret_ar_7.fit()
print(results_norm_ret_ar_7.summary())

## hasil: hasilnya sama dengan model return non-normalize, yang beda hanya nilai kontanta
## dapat disimpulkan bahwa normalize tidak mempengaruhi model selection
## dari kesimpulan diatas, model terbaik yaitu model dengan lags 6

### Analisis Residual of Normalize Returns ###
df['res_norm_ret'] = results_norm_ret_ar_6.resid
print(df.res_norm_ret.mean())
print(df.res_norm_ret.var())

### DF Test untu stationary residual data ###
print(sts.adfuller(df.res_norm_ret))

### ACF Test untuk Residual ###
sgt.plot_acf(df.res_norm_ret, zero=False, lags=40)
plt.title("ACF of Residuals for Normalize Return Prices", size=20)
plt.show()

### Plotting Residual vs Prices ###
df.norm_ret.plot(figsize=(20,5))
plt.title("Normalize Return Prices", size=20)
plt.show()

df.res_norm_ret.plot(figsize=(20,5))
plt.title("Residual of Normalize return Prices", size=20)
plt.show()