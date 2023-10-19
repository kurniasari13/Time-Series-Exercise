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
#pct_change: mencari returns, dan mul: perkalian

#### DF Test ###
print(sts.adfuller(df.returns[1:]))
##hasil: z statistik lebih kecil dari alpha (reject Ho), artinya data is stationary 

#### model dari berbagai pasar saham di dunia dapat dibandingkan selama semua data di setiap model sudah dinormalize

### NORMALIZE OF RETURNS ###
## di TS: normalize dengan cara menghitung percentage harga terhadap harga awal atau harga benchmark
bench_ret = df.returns.iloc[1]
df['norm_ret'] = df.returns.div(bench_ret).mul(100)

### Test DF untuk stationary ###
print(sts.adfuller(df.norm_ret[1:]))
##hasil: data  normalize return ternyata stationary sama dengan data return (normalize does not effect stationary)

### ACF Plot for Normalize of Returns ###
sgt.plot_acf(df.norm_ret[1:], zero=False, lags=40)
plt.title("ACF of Normalize Returns", size=23)
plt.show()
## hasil: koeff ACF sama dengan ACF returns sebelum normalize

### MA Models ###
model_norm_ret_ma_8 = ARIMA(df.norm_ret[1:], order=(0,0,8))
results_norm_ret_ma_8 = model_norm_ret_ma_8.fit()
print(results_norm_ret_ma_8.summary())
## koeff dan p value sama dengan model tanpa normalze, artinya normalize tidak mempengaruhi model selection
## yang berbeda hanya koeff konstanta

### Analisis Residual ###
df['res_norm_ret_ma_8'] = results_norm_ret_ma_8.resid[1:]

df.res_norm_ret_ma_8[1:].plot(figsize=(20,5))
plt.title("Residual of Normalize Returns", size=23)
plt.show()

sgt.plot_acf(df.res_norm_ret_ma_8[2:], zero=False, lags=40)
plt.title("ACF of Residuals for Normalize Returns", size=23)
plt.show()
## hasil: dari ACF plot, data disini resemble white noise, so our model is supposedly correct