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

### Create Returns ###
df['returns'] = df.market_value.pct_change(1)*100

### Create Squared Returns ###
df['sq_returns'] = df.returns.mul(df.returns)

### Create Plot Returns vs Square Returns ###
df.returns.plot(figsize=(20,5))
plt.title("Returns", size=23)
plt.show()

df.sq_returns.plot(figsize=(20,5))
plt.title("Volatility", size=23)
plt.show()
## hasil: the periods of low positive and negative returns are expressed with low volatility, while those with sharp jumps or drops have high uncertainly

### Create Plot PACF Returns vs Square Returns ###
sgt.plot_pacf(df.returns[1:], lags=40, alpha=0.05, zero=False, method=('ols'))
plt.title("PACF of Returns", size=20)
plt.show()
## hasil: beberapa lags pertama signifikan, but we can not directly use the rule of thump we just introduces
## we should still have a look at the squared values to get an idea about which lags are significant
## even though we are not getting the suggested number of lags, we are still gaining valuable insight on how the data performs

sgt.plot_pacf(df.sq_returns[1:], lags=40, alpha=0.05, zero=False, method=('ols'))
plt.title("PACF of Squared Returns", size=20)
plt.show()
## hasil: we can see that the fisrt six lags are significant with the first five yielding coefficients between 0.15 - 0.25, 
# such high significant values of partial autocorrelation among the first few lags suggests that there tend to be short term trends in variance
## there tend to be short term trends in variance, another way to think about this is clustering where we have periods of high variation followed by periods of hugh variations, 
# as well as periods of low variation followed by periods of low variation, this is eactly what the volatility characteristic stated earlier (naik perlahan/turun perlahan) 
## so our data set makes perfect sense

### The ARCH Model(1) ###
from arch import arch_model

model_arch_1 = arch_model(df.returns[1:], mean="Constant", vol= "ARCH", p=1)
results_arch_1 = model_arch_1.fit(update_freq=5)
print(results_arch_1.summary())

##if we dont pass any values to the other arguments (of the fit method), this is type of model which is different from the ARCH we intended to use
## iteration yang awalnya 13 menjadi 6, the specification were loose enough, our model is light and does not take too long to compute
##  R squared is a measurement of explanatory variation away from the mean
#  jika constant mean, it makes little sense to expect this model to explain that deviation well
# if the residual are simply a version of the original data set, ehre every value is decreased by a constant, then there will be no actual variance, there is nothing to explain, so r square is zero

## hasil: df ada tiga karena menggunakan lags 1 dan constant mean 
## e-x = /(10 * x), e+x = *(10 * x)
## hasil: semua koeff signifikan, log likelihood more higher than ARIMAX family
## meskipun arch menjeadi model terbaik so far, it is crucial to remember that the arch model only be used to predict future variants rather than future returns/values of variables
## thus we can use it to determine if we expect to see stability in the market, but not predict if the prices would go up or down
## mean bisa pake AR dan bisa menentukan lags, or practice setting different probability distributions for the error terms

model_arch_1 = arch_model(df.returns[1:], mean="AR", lags=[2,3,6], vol= "ARCH", p=1, dist="ged")
results_arch_1 = model_arch_1.fit(update_freq=5)
print(results_arch_1.summary())

model_arch_2 = arch_model(df.returns[1:], mean="Constant", vol= "ARCH", p=2)
results_arch_2 = model_arch_2.fit(update_freq=5)
print(results_arch_2.summary())
## hasil: iteration naik karena model harus menghitung additional coefficient
## log likelihood naik dan AIC turun, semua koeff signifikan 

model_arch_3 = arch_model(df.returns[1:], mean="Constant", vol= "ARCH", p=3)
results_arch_3 = model_arch_3.fit(update_freq=5)
print(results_arch_3.summary())
## log likelihood naik dan AIC turun, semua koeff signifikan 
### trial and error ini coba sampai lags 13 dimana ada salah satu koeff tisak signifikan
## terjadi diminishing marginal log likelihood setiap penambahan lags
## the more past squared residuals we take into account, the less important each additional one becomes
