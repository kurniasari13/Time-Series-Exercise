import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 
sns.set()

### Preprocessing Time Series ###

raw_csv_data = pd.read_csv('D:/DATA ANALYST/belajar_python/TIME SERIES/Index2018.csv')
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)
df_comp.set_index("date", inplace=True)
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method='ffill')

df_comp['market_value'] = df_comp.spx
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### White Noise ###

wn = np.random.normal(loc= df.market_value.mean(), scale= df.market_value.std(), size=len(df))
df['wn'] = wn
print(df.describe())

### Compare Plot between White Noise dengan Actual Data ###

df.wn.plot(figsize=(20,5))
plt.title("White Noise Time Series", size=24)
plt.show()

df.market_value.plot(figsize=(20,5))
plt.title("S&P Prices", size=24)
plt.ylim(0,2300)
plt.show()
# hasil: plot untuk actual data memiliki far smaller jump per periode karena datanya non-random
# sehingga kita perlu tau pattern untuk mendapatkan accurate forecast

### Random Walk ###

#hanya membandingkan data yng ada random walk dengan yang tidak ada
rw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/TIME SERIES/RandWalk.csv')
rw = rw_data.copy()

rw.date = pd.to_datetime(rw.date, dayfirst=True)
rw.set_index("date", inplace=True)
rw = rw.asfreq('b')
print(rw.describe())

df['rw'] = rw.price
print(df.head())

df.rw.plot(figsize=(20,5))
plt.title("Random Walk", size=24)
plt.show()

df.rw.plot(figsize=(20,5))
df.market_value.plot()
plt.title("Random Walk vs S&P Prices", size=24) 
plt.show() 
# hasil: both have small variations between consecutive time period (jarak harga satu hari dengan hari lainnya gk jauh beda)
#hasil: both have cyclucal increase and decrease in short periods of time

### Mengukur Stationarity ###

print(sts.adfuller(df.market_value))
#hasil: test statistik lebih besar dari crirical value di semua alpha
#hasil: do not reject Ho: we do not find sufficient evidence of stationary in data set

print(sts.adfuller(df.wn))
print(sts.adfuller(df.rw))

### Seasonality ###
s_dec_additive = seasonal_decompose(df.market_value, model="additive")
s_dec_additive.plot()
plt.show()

#hasil: trend mirip observed value karena decomposition function uses the previous period values as a trendsetter
#dengan pattern: current period prices are the best predictor for the next prices
#jika kita observe seasonal pattern, we will have other price s as better predictors
#contoh season: jika harga lebih tinggi pada awal bulan, maka lebih baik menggunakan values dari 30 hari yang lalu daripada 1 hari yang lalu
#trend decomposition menjelaskan most variablity of the data
#seasonal: bentuknya persegi panjang karena nilainya secara konstan oscillating back and forth and figure size is too small
#seasonal: mandar mandir antara -0.2 dan 1 in any period, artinya tidak ada concrete cyclical pattern 
#residual yang besar karena ekonomi yg tidak stabil di perioe tertentu
#overall, no seasonality in the data

#double cek lagi dengan pendekatan yang berbeda
s_dec_multiplicative = seasonal_decompose(df.market_value, model="multiplicative")
s_dec_multiplicative.plot()
plt.show()

### Autocorrelation Function or ACF ###
sgt.plot_acf(df.market_value, lags=40, zero=False)
plt.title("ACF S&P", size=24)
plt.show()

sgt.plot_acf(df.wn, lags=40, zero=False)
plt.title(" ACF White Noise", size=24)
plt.show()

sgt.plot_acf(df.rw, lags=40, zero=False)
plt.title(" ACF Random Walk", size=24)
plt.show()

### Partial Autocorrelation Function or PACF ###
sgt.plot_pacf(df.market_value, lags=40, zero=False, method=('ols'))
plt.title("PACF S&P", size=24)
plt.show()

sgt.plot_pacf(df.wn, lags=40, zero=False, method=('ols'))
plt.title("PACF White Noise", size=24)
plt.show()

sgt.plot_pacf(df.rw, lags=40, zero=False, method=('ols'))
plt.title("PACF Random Walks", size=24)
plt.show()