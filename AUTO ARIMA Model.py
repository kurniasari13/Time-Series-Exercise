import numpy as np
import pandas as pd 
import scipy
import statsmodels.api as sm 
import matplotlib as plt 
import seaborn as sns 
import sklearn
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import yfinance
import warnings 
warnings.filterwarnings("ignore")
sns.set()

### Loading the Data ###
raw_data = yfinance.download(tickers="^GSPC ^FTSE ^N225 ^GDAXI", start="1994-01-07", end="2018-01-29", interval="1d", group_by='ticker', auto_adjust=True, treads=True)

df_comp = raw_data.copy()
df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]

df_comp = df_comp.iloc[1:]
del df_comp['^N225']
del df_comp['^GSPC']
del df_comp['^GDAXI'] 
del df_comp['^FTSE']
df_comp = df_comp.asfreq('b')
df_comp = df_comp.fillna(method='ffill')

### Creating Returns ###
df_comp['ret_spx'] = df_comp.spx.pct_change(1)*100
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1)*100
df_comp['ret_dax'] = df_comp.dax.pct_change(1)*100
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1)*100

### Splitting the Data ###
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### Fitting the Model ###
from pmdarima.arima import auto_arima

### Defaulth auto arima
model_auto = auto_arima(df.ret_ftse[1:])
## the defaulth best ARIMA model, these defaults are notalways optimal or reasonable
print(model_auto)
print(model_auto.summary())
## hasil: sarimax(4,0,5), sarimax is the most general model (seasonality, auto regression, integration, moving average, and exogenous)
## hasil: ARMA(4, 5) was not our preferred model of choice when we conducted our initial manual analysis karena beberapa koeff yang tidak signifikan  
## few quick comments: the rules of model selection are rather "rules of thump" than "fix"
## additionally, the auto arima only considers a single feature, the AIC
#  we can have easily overfitting while going through the models in our previous sections
# the default arguments of the method restrict the number of AR and MA components

### Basic auto arima
model_auto_basic = auto_arima(df.ret_ftse[1:], exogenous=df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m=5, max_order=None, max_p=7, max_q=7, max_d=2, 
                                                max_P=4, max_Q=4, max_D=2, maxiter=50, alpha=0.05, n_jobs=-1, trend='ct')

### Advance auto arima
model_auto_advance = auto_arima(df_comp.ret_ftse[1:], exogenous=df_comp[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m=5, max_order=None, max_p=7, max_q=7, max_d=2, 
                    max_P=4, max_Q=4, max_D=2, maxiter=50, alpha=0.05, n_jobs=-1, trend='ct', infornation_criterion="oob", out_of_sample_size=int(len(df_comp)*0.02))
print(model_auto_advance.summary())

## jika menggunakan out of bag or oob, we need to separate the data into a sample and out of sample sets 
### since we are letting the method validate the results, we no longer need to plug in the training set, but rather the complete data set
## hasill: lebih sedikit lags untuk AR dan MA non-seasonal
## drift koeff: represents the linier trend koeff, we expected to see a constant as well as linier term and their koeff are the intercept and the drift  
# AIC menurun  
