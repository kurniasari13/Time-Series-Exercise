import numpy as np
import pandas as pd 
import scipy
import statsmodels.api as sm 
import matplotlib as plt 
import seaborn as sns 
import sklearn
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from arch import arch_model
import yfinance
import warnings 
warnings.filterwarnings("ignore")
sns.set()

### Loading the Data ###
raw_data = yfinance.download(tickers="^GSPC ^FTSE ^N225 ^GDAXI", start="1994-01-07", end="2018-01-29", 
            interval="1d", group_by='ticker', auto_adjust=True, treads=True)

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

df_comp['norm_ret_spx'] = df_comp.ret_spx.div(df_comp.ret_spx[1])*100
df_comp['norm_ret_ftse'] = df_comp.ret_ftse.div(df_comp.ret_ftse[1])*100
df_comp['norm_ret_dax'] = df_comp.ret_dax.div(df_comp.ret_dax[1])*100
df_comp['norm_ret_nikkei'] = df_comp.ret_nikkei.div(df_comp.ret_nikkei[1])*100


### Splitting the Data ###
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### Fitting a Model AR(1) ###
model_ar = ARIMA(df.ftse, order=(1,0,0))
results_ar = model_ar.fit()

### Simple Forecasting  AR(1) ###
print(df.tail())

start_date = "2014-07-15"
end_date = "2015-01-01"
#end_date = "2019-10-23"
## make sure the start and end dates are business days, otherwise the code will result in an error

df_pred = results_ar.predict(start= start_date, end= end_date)

df_pred[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions AR(1) vs Actual Prices", size=24)
plt.show()
## we see a constantly decreasing line, which is not at all realistic in practice
## jika hasil selalu menurun, maka harga hari ini selalu tinggi dan harga besok hari sellau rendah
## seller want to sell today and buyer want to buy tomorrow
## penyebab: the issue comes from our model of choice, the predictions are only based on the constant and the prices from the previous period
## every new value is just a fraction of the previous one put on top of the constant term

## saat dibandingkan dengan actual value, actual price cyclically jump up and down around the values we are especting 
## dalam jangka panjang maupun jangka pendek model AR is bad estimator

### Forecasting AR(1) Returns ###
end_date = "2015-01-01"

model_ret_ar = ARIMA(df.ret_ftse[1:], order=(1,0,0))
results_ret_ar = model_ret_ar.fit()

df_pred_ar = results_ret_ar.predict(start= start_date, end= end_date)

df_pred_ar[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions AR(1) vs Actual Returns", size=24)
plt.show()
## hasil: garis predction to be a constant line at the zero, this means that our model makes no prediction since it assume all future returns will be zero or extremely close to it
## this is tell us that the koeff for the past values as well as the values themselves must have low absolute values 
print(results_ret_ar.summary())
## koeff tidak signifikan, it is makes sense for the model predictions to barely move over time

end_date = "2014-08-23"

model_ret_ar = ARIMA(df.ret_ftse[1:], order=(5,0,0))
results_ret_ar = model_ret_ar.fit()

df_pred_ar = results_ret_ar.predict(start= start_date, end= end_date)

df_pred_ar[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions AR(1) vs Actual Returns", size=24)
plt.show()
## saat order AR dinaikkan menjadi 5, we will see some very small shift among the initial forecasting, but the curve still flattens out very quickly since forecasted values become smaller and smaller in absolute value
#  additionally, we noticed that for every single period where the value shift, they are moving in the opposite direction of where they should be going, in the first period they should be going up but they are decreasing, then the opposite happen for the second forecast and so on
#  increasing the orer makes little difference in the forecasting capabilities of the model 

### Forecasting MA(1) returns ###
end_date = "2015-01-01"

model_ret_ma = ARIMA(df.ret_ftse[1:], order=(0,0,1))
results_ret_ma = model_ret_ma.fit()

df_pred_ma = results_ret_ma.predict(start= start_date, end= end_date)

df_pred_ma[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions MA(1) vs Actual Returns", size=24)
plt.show()
## we see a similar pattern where the predictions are jut a flat line, very close to zero 
print(df_pred_ma.head()) #untuk melihat values apakah 0 bisa dengan cara zoom in or call head method
## all the values after the first period dont change, it is because the MA koeff so small we are essensially only left with the constant after the initial few time periods

### Forecasting ARMA(1,1) returns ###
end_date = "2015-01-01"

model_ret_arma = ARIMA(df.ret_ftse[1:], order=(1,0,1))
results_ret_arma = model_ret_arma.fit()

df_pred_arma = results_ret_arma.predict(start= start_date, end= end_date)

df_pred_arma[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions ARMA(1,1) vs Actual Returns", size=24)
plt.show()
## prediksinya berbentuk garis lurus seperti AR dan MA
print(df_pred_arma.head())
## we have values consistently decreasing very slowly
print(df_pred_arma.tail())
## the curve flatten out (nilainya sama semua)
## the ARMA provides more reasonable predictions as they dont just die off immediately, at the same time, the shift are always in the same direction down
# this means that we will still be having constanly decreasing returns, which is not very realistic

### Forecasting ARMAX(1,1) returns ###
end_date = "2015-01-01"

model_ret_armax = ARIMA(df.ret_ftse[1:], exog=df[["ret_spx", "ret_dax", "ret_nikkei"]][1:], order=(1,0,1))
results_ret_armax = model_ret_armax.fit()

df_pred_armax = results_ret_armax.predict(start= start_date, end= end_date, exog=df[["ret_spx", "ret_dax", "ret_nikkei"]][start_date:end_date])

df_pred_armax[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions ARMAX(1,1) vs Actual Returns", size=24)
plt.show()
## finaly we see that our predictions match the true values a lot more closely than ever before, di beberapa bagian ada yang overestimates dan underestimates, overall it follow the data really well
## this indicates that including outside factors like other market value indexes improve the predictive power dramatically
## reguler arma model can not incorporate any outside real world effects

### Forecasting ARIMA(1,1,1) Prices ###
end_date = "2015-01-01"

model_arima = ARIMA(df.ftse[1:], order=(1,1,1))
results_arima = model_arima.fit()

df_pred_arima = results_arima.predict(start= start_date, end= end_date)

df_pred_arima[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions ARMAX(1,1) vs Actual Prices", size=24)
plt.show()

### Forecasting ARIMAX(1,1,1) Prices ###
end_date = "2015-01-01"

model_arimax = ARIMA(df.ftse[1:], exog=df[["spx", "dax", "nikkei"]][1:], order=(1,1,1))
results_arimax = model_arimax.fit()

df_pred_arimax = results_arimax.predict(start= start_date, end= end_date, exog=df[["spx", "dax", "nikkei"]][start_date:end_date])

df_pred_arimax[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions ARIMAX(1,1) vs Actual Prices", size=24)
plt.show()

### Forecasting SARMA() Prices ###
end_date = "2015-01-01"

model_sarma = SARIMAX(df.ret_ftse[1:], order=(3,0,4), seasonal_order=(3,0,2,5))
results_sarma = model_sarma.fit()

df_pred_sarma = results_sarma.predict(start= start_date, end= end_date)

df_pred_sarma[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions SARMA vs Actual Returns", size=24)
plt.show()
## we see that most values are close to zero, but the prediction curve is wiggly instead of being constantly decreasing (naik turun kurvanya)
## the magnitude is far too small compared to the actual returns, sarma model anticipate greater stability than the MAX model we examined earlier, but the accuracy is lacking

### Forecasting SARIMAX() Prices ###
end_date = "2015-01-01"

model_sarimax = SARIMAX(df.ret_ftse[1:], exog=df[["ret_spx", "ret_dax", "ret_nikkei"]][1:],
                order=(3,0,4), seasonal_order=(3,0,2,5))
results_sarimax = model_sarimax.fit()

df_pred_sarimax = results_sarimax.predict(start= start_date, end= end_date, exog=df[["ret_spx", "ret_dax", "ret_nikkei"]][start_date:end_date])

df_pred_sarimax[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions SARIMAX vs Actual Returns", size=24)
plt.show()
## the curve follow the data very closely, a similar improvement occured when we went from ARMA to the ARMAX, so adding exogenous variables drastically changes our predictions
## therefore, the best estimates seem to be ones where we are adding exogenous variables into the mix

### Forecasting Auto ARIMA ###
model_auto = auto_arima(df.ret_ftse[1:])

df_auto_pred = pd.DataFrame(model_auto.predict(n_periods=len(df_test[start_date:end_date])), index = df_test[start_date:end_date].index)

df_auto_pred.plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions Auto ARIMA  vs Actual Returns", size=24)
plt.show()
## berbeda dengan statmodel yang menggunakan argument start and dates saat memprediksi, auto arima menggunakan n_periods which indicates how many elements were predicting
## the predictions dont really match the curve as closely as the SARMAX model, this is the defaulth best model, so we should not be too bothered about the accuracy

model_auto = auto_arima(df.ret_ftse[1:], exogenous=df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m=5, max_p=5, max_q=5, max_P=5, max_Q=5)

df_auto_pred = pd.DataFrame(model_auto.predict(n_periods=len(df_test[start_date:end_date])), exogenous=df[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date],index = df_test[start_date:end_date].index)

df_auto_pred.plot(figsize=(20,5), color="red")
df_test.ret_ftse[start_date:end_date].plot(color="blue")
plt.title("Predictions Auto ARIMA  vs Actual Returns", size=24)
plt.show()
## the curve matches the testing set far better, as long as we provide the auto arima method with enough information, it is bound to give us a reasonable solution
## we notice how all the models that perform well rely on exogenous variables, however those are hardly ever available when we want to make long term prediction
## our max models rely on outside data much more than they do on past values or past errors

### Comparing All the Models (Returns) ###
end_date = "2015-01-01"
df_pred_ar[start_date:end_date].plot(figsize=(20,10), color="yellow")
df_pred_ma[start_date:end_date].plot( color="pink")
df_pred_arma[start_date:end_date].plot( color="cyan")
df_pred_armax[start_date:end_date].plot( color="green")
df_pred_sarma[start_date:end_date].plot( color="magenta")
df_pred_sarimax[start_date:end_date].plot( color="red")
df_test.ret_ftse[start_date:end_date].plot(color= "blue")
plt.legend(['AR', 'MA', 'ARMA', 'ARMAX', 'SARMA', 'SARIMAX'])
plt.title("ALll the Models", size= 23)
plt.show()
## dari plot diatas diketahui bahwa model SARIMAX model terbaik so far

### Forecatong GARCH Model ###
mod_garch = arch_model(df_comp.ret_ftse[1:], vol="GARCH", p =1, q=1, mean="constant", dist="Normal")
res_garch = mod_garch.fit(last_obs=start_date, update_freq = 10)

pred_garch = rest_garch.forecast(horizon=1, align='target')
## horizon: how many observations we want our model to predict for each date
## align: determines whether we match the value with the data the prediction is made on, or the one it is supposed to represent

pred_garch.residual_variance[start_date:].plot(figsize=(20,5), color="red", zorder=2)
df_test.ret_ftse().plot(color="blue", zorder=1)
plt.title("Volatility Predictions", size=24)
plt.show()
## we see the curve that fluctuates up and down and show no obvious pattern, our volatility prediction are not linier like ARMA model, but fluctuate based on the past conditional variance
## our model does a decent job predicting when shocks will happen and does very well in determining periods of consecutive high and low volatility

pred_garch = res_garch.forecast(horizon=100, align='target')
print(pred_garch.residual_variance[-1:])

### Multivariate Regression Model ###
from statsmodels.tsa.api import VAR
df_ret= df[['ret_spx', 'ret_dax', 'ret_ftse', 'ret_nikkei']][1:]

model_var_ret = VAR(df_ret)
model_var_ret.select_order(20)
results_var_ret = model_var_ret.fit(ic='aic')
print(results_var_ret.summary())

lag_order_ret = results_var_ret.k_ar
var_pred_ret = results_var_ret.forecast(df_ret.value[-lag_order_ret:], len(df_test[start_date:end_date]))
df_ret_pred = pd.DataFrame(data=var_pred_ret, index=df_test[start_date:end_date].index, columns=df_test[start_date:end_date].columns[4:8])

df_ret_pred.ret_nikkei[start_date:end_date].plot(figsize=(20,5), color="red")
df_test.ret_nikkei[start_date:end_date].plot(color="blue")
plt.title("Real vs Prediction", size=24)
plt.show()
### the curve is flatten after several periods, we are only including past values for these additional time series, we are not including the extreme valuable input of today's values into the mix, the model performs just like an expanded AR model

results_var_ret.plot_forecast(1000)
plt.show()
