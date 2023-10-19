import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
#import statsmodels.api as sm
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts
from arch import arch_model
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

### Create Returns ###
df['returns'] = df.market_value.pct_change(1)*100

## GARCH MODEL (1,1) ###
model_garch_1_1 = arch_model(df.returns[1:], mean="Constant", vol="GARCH", p=1, q=1)
results_arch_1_1 = model_garch_1_1.fit(update_freq=5)
print(results_arch_1_1.summary())
##hasil: adding a single past variance gives more predictive power than 11 squared residuals
##koeff semua signifikan, this model instantly become front runner for measuring volatility

### Fitting The Higher GARCH Model ###
model_garch_1_1 = arch_model(df.returns[1:], mean="Constant", vol="GARCH", p=1, q=2)
results_arch_1_1 = model_garch_1_1.fit(update_freq=5)
print(results_arch_1_1.summary())
## koeff untuk lags 2variance squared is 1, its mean full multicollinearity, all the explanatory power of the conditional variance 2 periods ago is alreasy captured by the variance from last period

model_garch_1_1 = arch_model(df.returns[1:], mean="Constant", vol="GARCH", p=1, q=3)
results_arch_1_1 = model_garch_1_1.fit(update_freq=5)
print(results_arch_1_1.summary())
## koeff lags 2 dan 3 tidak signifikan

model_garch_1_1 = arch_model(df.returns[1:], mean="Constant", vol="GARCH", p=2, q=1)
results_arch_1_1 = model_garch_1_1.fit(update_freq=5)
print(results_arch_1_1.summary())
## hasil: even though we dont get a p values of 1, we can still see the additional koeff is not significant

model_garch_1_1 = arch_model(df.returns[1:], mean="Constant", vol="GARCH", p=3, q=1)
results_arch_1_1 = model_garch_1_1.fit(update_freq=5)
print(results_arch_1_1.summary())
## koeff arch model lag 2 dan 3 tidak signifikan, this model fails as well

## conclution: the garch(1,1) is the best odel for measuring volatility of returns, no need to rely on overly complicated models
