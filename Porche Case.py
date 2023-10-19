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
from pmdarima.arima import OCSBTest
from arch import arch_model
import yfinance
import warnings 
warnings.filterwarnings("ignore")
sns.set()

### Import The Data ###
raw_data = yfinance.download(tickers = "VOW3.DE, PAH3.DE, BMW.DE", interval="1d", group_by='ticker', auto_adjust=True, treads=True)
df = raw_data.copy()

### Defining Key Dates ###
# starting date
start_date = '2009-04-05'

# first official announcement 49.9%
ann_1 = '2009-12-09'

# second official announcement 50.1%
ann_2 = '2012-07-05'

# ending date
end_date = '2014-01-01'

# diesel gate skandal 
d_gate = '2015-09-20'

### Preprocessing The Data ###
# extracting closing prices
df['vol'] = df['VOW3.DE'].Close
df['por'] = df['PAH3.DE'].Close
df['bmw'] = df['BMW.DE'].Close

# creating returns
df['ret_vol'] = df['vol'].pct_change(1).mul(100)
df['ret_por'] = df['por'].pct_change(1).mul(100)
df['ret_bmw'] = df['bmw'].pct_change(1).mul(100)

# creating squared returns
df['sq_vol'] = df.ret_vol.mul(df.ret_vol)
df['sq_por'] = df.ret_pol.mul(df.ret_pol)
df['sq_bmw'] = df.ret_bmw.mul(df.ret_bmw)

# extracting volume
df['q_vol'] = df['VOW3.DE'].Volume
df['q_por'] = df['PAH3.DE'].Volume
df['q_bmw'] = df['BMW.DE'].Volume
## the more a certain stock is being traded on a given day, the more likely it is for its price to fluctuate

# assigning the frequency and filling na values
df = df.asfreq('b')
df = df.fillna(method='bfill')

# removing surplus data
del df['VOW3.DE']
del df['PAH3.DE']
del df['BMW.DE']

## we usually split the data into a training and a testing set
## we dont need split, because we examining a specific event in time, rather than trying to predict the future

### Plotting the Prices ###
df['vol'][start_date:end_date].plot(figsize=(20,8), color="blue")
df['por'][start_date:end_date].plot(color="green")
df['bmw'][start_date:end_date].plot(color="gold")
plt.show()
## we see the similarity in the way they move which indicates trends of the entire automobile industry market   
## however VW prices shift a lot more in magnitude after the third quarter of 2009

# shades of blue: "#33B8FF" "#1E7EB2" "#0E3A52"
# shades of green: "#49FF3A" "#2FAB25" "#225414"
# shades of gold: "#FEB628" "#BA861F" "#7C5913"

df['vol'][start_date:ann_1].plot(figsize=(20,8), color="#33B8FF")
df['por'][start_date:ann_1].plot(color="#49FF3A")
df['bmw'][start_date:ann_1].plot(color="#FEB628")

df['vol'][start_date:ann_2].plot(color="#1E7EB2")
df['por'][start_date:ann_2].plot(color="#2FAB25")
df['bmw'][start_date:ann_2].plot(color="#BA861F")

df['vol'][start_date:end_date].plot(color="#0E3A52")
df['por'][start_date:end_date].plot(color="#225414")
df['bmw'][start_date:end_date].plot(color="#7C5913")

plt.legend(['Volkswagen', 'Porsche', 'BMW'])
plt.show()

## the first announcement, we see that the two stoks VM and Porsche move in a similiar fashion, volkswagen numbers seem to roughkt twice as high, the the gap between the two seems to grow bigger and bigger
## if we look at BMW numbers we will see that they resemble the porsche ones much closely than the VW ones

### Correlation ###
print('Correlation among among manufacturers from' + str(start_date) + 'to' + str(end_date) + '\n')
print('Volkswagen and Porsche correlation: \t' + str(df['vol'][start_date:end_date].corr(df[por][start_date:end_date])))
print('Volkswagen and BMW correlation: \t' + str(df['vol'][start_date:end_date].corr(df['bmw'][start_date:end_date])))
print('Porsche and BMW correlation: \t\t' + str(df['por'][start_date:end_date].corr(df['bmw'][start_date:end_date])))

## we get higher correlation between BMW and Volkswagen prices than  between Porsche and VW
# this suggest that VM moves in a similar way to the market benchmark, meskipun in much higher values
## the way VW owns 100% of porsche the end of this time interval, shoukd not the correlation be much higher?
# to understand what is really going on, we do this by examining the three different interval one by one, tsrating from the earliest one
# to only change we make requires changing the end og the interval variabke

print('Correlation among among manufacturers from' + str(start_date) + 'to' + str(ann_1) + '\n')
print('Volkswagen and Porsche correlation: \t' + str(df['vol'][start_date:ann_1].corr(df[por][start_date:ann_1])))
print('Volkswagen and BMW correlation: \t' + str(df['vol'][start_date:ann_1].corr(df['bmw'][start_date:ann_1])))
print('Porsche and BMW correlation: \t\t' + str(df['por'][start_date:ann_1].corr(df['bmw'][start_date:ann_1])))
## we see much lower correlation all around, porche and BMW have lowest values, this means Porsche and BMW stock prices were not to similar before the start buyout

print('Correlation among among manufacturers from' + str(ann_1) + 'to' + str(ann_2) + '\n')
print('Volkswagen and Porsche correlation: \t' + str(df['vol'][ann_1:ann_2].corr(df[por][ann_1:ann_2])))
print('Volkswagen and BMW correlation: \t' + str(df['vol'][ann_1:ann_2].corr(df['bmw'][ann_1:ann_2])))
print('Porsche and BMW correlation: \t\t' + str(df['por'][ann_1:ann_2].corr(df['bmw'][ann_1:ann_2])))
## the new plot show an increase across all correlation, the interesting output here is almost 98% correlation between VM and BMW

print('Correlation among among manufacturers from' + str(ann_2) + 'to' + str(end_date) + '\n')
print('Volkswagen and Porsche correlation: \t' + str(df['vol'][ann_2:end_date].corr(df[por][ann_2:end_date])))
print('Volkswagen and BMW correlation: \t' + str(df['vol'][ann_2:end_date].corr(df['bmw'][ann_2:end_date])))
print('Porsche and BMW correlation: \t\t' + str(df['por'][ann_2:end_date].corr(df['bmw'][ann_2:end_date])))

## we see a higher correlation between VW and Porsche than between VW and BMW, this goes to show that the two are truly being recognized as a single entity and the price of one directly effect the other
## however he highedt correlation is the one between BMW and Porsche which is interesting considering they have no obvious link between them 

print('Correlation among among manufacturers from' + str(end_date) + 'to' + str(df.index[-1]) + '\n')
print('Volkswagen and Porsche correlation: \t' + str(df['vol'][end_date:].corr(df[por][end_date:])))
print('Volkswagen and BMW correlation: \t' + str(df['vol'][end_date:].corr(df['bmw'][end_date:])))
print('Porsche and BMW correlation: \t\t' + str(df['por'][end_date:].corr(df['bmw'][end_date:])))

## we see that the correlation between Porsche and VW remain highm but the other two have fallen drastically 
## if there another boom in the VW prices would once again lead the market with it, but this does not appear to be the case
## what actually happen? the diesel gate scandal, which picked up steam in september 2015, this event hit volkswagen much harder than BMW and Porsche 

### Best Fitting Models ###
## VW
mod_pr_pre_vol = auto_arima(df.vol[start_date:ann_1], exogenous=df[['por', 'bmw']][start_date:ann_1], m=5, max_p=5, max_q=5)
mod_pr_btn_vol = auto_arima(df.vol[ann_1:ann_2], exogenous=df[['por', 'bmw']][ann_1:ann_2], m=5, max_p=5, max_q=5)
mod_pr_post_vol = auto_arima(df.vol[ann_2:end_date], exogenous=df[['por', 'bmw']][ann_2:end_date], m=5, max_p=5, max_q=5)

print(mod_pr_pre_vol.summary())
print(mod_pr_btn_vol.summary())
print(mod_pr_post_vol.summary())

## sebelum pengumuman 1, the best model is AR(1) with two exog variables, all koeff signifikan kecuali intercept, so this is indeed a good fit, koeff ar 1 sangat mendekati 1, this means we are sticking really close to the value of the last period with very little deviation 
## setelah pengumuman 1, the best model is ARIMAX (1,1,1) with two exogenous variables, the optimal models is interated one yang artinya data original tdk sationary, the model finds past residuals to have explanatory power, this indicates that the real announcement actually changed the trend for VW prices
## setelah pengumuman 2, best model is ARIMAX(0,1,0) with two exogenous variables, koeff Porsche naik, this is occurs because the two automobile manufacture are now a single entity, the price of porsche today will be a more accurate estimator than the price of VW yesterday (sehingga tidak pakai MA), therefore new information seems to have a higher impact on trends compared to past patterns

mod_pr_pre_por = auto_arima(df.por[start_date:ann_1], exogenous=df[['vol', 'bmw']][start_date:ann_1], m=5, max_p=5, max_q=5)
mod_pr_btn_por= auto_arima(df.por[ann_1:ann_2], exogenous=df[['vol', 'bmw']][ann_1:ann_2], m=5, max_p=5, max_q=5)
mod_pr_post_por = auto_arima(df.por[ann_2:end_date], exogenous=df[['vol', 'bmw']][ann_2:end_date], m=5, max_p=5, max_q=5)

print(mod_pr_pre_por.summary())
print(mod_pr_btn_por.summary())
print(mod_pr_post_por.summary())
## sebelum pengumuman, AR(2) model with two exogenous variables
## setelah pengumaman pertama, ARIMA(1,1,0) is best model, koeff ar turun dibandingkan model sebelumnya because we are using integrated values here, and an additional explanation comes from the rise of the VW koeff which now has a bigger impact
## setelah pengumuman kedua, SARIMAX(0,1,1)(0,0,1,5) with two endogenous variables is best model, interestingly enough this time the model discovers a seasonal pattern which we have not encountered before, these trends are effected more by current events rather than pre existing patterns

## we just examined two cases where see major trends changes occuring after new infoemation is introduced 
## this event change the correlation and the general patterns within a data set

### Predictions for the Future ###
## without exogenus variables
model_auto_pred_pr = auto_arima(df.vol[start_date:ann_1], m=5, max_q=5, max_p=5, max_P=5, max_Q=5, trend="ct")

df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods=len(df[ann_1:ann_2])), index=df[ann_1:ann_2].index)

df_auto_pred_pr[ann_1:ann_2].plot(figsize=(20,5), color="red")
df.vol[ann_1:ann_2].plot(color="blue")
plt.title("VW Predictions (No Exog) vs Real Data", size=24)
plt.show()

## at first glance, the predictions seem very good over the first three or four months but then start to die off faster and faster while the actual prices steadly bounce right up
## this indicated that the announcement did nt make a great shift in price trends for VW initially, that can be attributed to the market lags where new information as well as policy changes take a while to respond appropriately
## we noticed that the forecasted prices are going down below zero which is impossible, this should be a huge red flag showing that this trend is not sustainable long term

## lest  zoom in
df_auto_pred_pr[ann_1:'2010-03-01'].plot(figsize=(20,5), color="red")
df.vol[ann_1:'2010-03-01'].plot(color="blue")
plt.title("VW Predictions (No Exog) vs Real Data (short term)", size=24)
plt.show()

## we see predictions were not really close day by day, but captured the general trend over the entire period
## this means the predictions would only be useful if we were trying to ballpark the quarterly performance of prices 

##predict with exogenous variables
# only Porsche
model_auto_pred_pr = auto_arima(df.vol[start_date:ann_1], exogenous=df[['por']][start_date:ann_1], m=5, max_q=5, max_p=5, max_P=5, max_Q=5, trend="ct")

df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods=len(df[ann_1:ann_2])), exogenous=df[['por']][ann_1:ann_2], index=df[ann_1:ann_2].index)

df_auto_pred_pr[ann_1:ann_2].plot(figsize=(20,5), color="red")
df.vol[ann_1:ann_2].plot(color="blue")
plt.title("VW Predictions (Porsche as Exog) vs Real Data", size=24)
plt.show()
## we see a trend that does not simply die off exponentially after several months, instead values fluctuate within some reasonable range 
# however after some time we see that the general trend and the lower porsche prices lead our forecasts astray (sesat), therefore adding porsche prices improves predictions but not subtantially since we are moving far away from the actual values after Q3 2010 
# since porsche is partially owend by VW that point in time, using market bencmark would probably be the wiser move, so try switching the exogenous variable to BMW  

#only BMW
model_auto_pred_pr = auto_arima(df.vol[start_date:ann_1], exogenous=df[['bmw']][start_date:ann_1], m=5, max_q=5, max_p=5, max_P=5, max_Q=5, trend="ct")

df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods=len(df[ann_1:ann_2])), exogenous=df[['bmw']][ann_1:ann_2], index=df[ann_1:ann_2].index)

df_auto_pred_pr[ann_1:ann_2].plot(figsize=(20,5), color="red")
df.vol[ann_1:ann_2].plot(color="blue")
plt.title("VW Predictions (BMW as Exog) vs Real Data", size=24)
plt.show()

## we can see that the forecasts are finally lining up with the general trend over rhe entire period rather than only the firt few months, however we see that the changes our forecats makes are more conservative (moderate) for both rises and falls
## in some intervals failure to adjust a major decrease results in overestimation for a prolonged period
## the same holds trye for great jumps occuring in Q3 of 2010 where our forecasting indicates some expected rise but fails to truly capture its magnitude 
## overall, these prediction resemble the real movement of the VW equity much better than the two previous models

# both Porsche and BMW
model_auto_pred_pr = auto_arima(df.vol[start_date:ann_1], exogenous=df[['por', 'bmw']][start_date:ann_1], m=5, max_q=5, max_p=5, max_P=5, max_Q=5, trend="ct")

df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods=len(df[ann_1:ann_2])), exogenous=df[['por', 'bmw']][ann_1:ann_2], index=df[ann_1:ann_2].index)

df_auto_pred_pr[ann_1:ann_2].plot(figsize=(20,5), color="red")
df.vol[ann_1:ann_2].plot(color="blue")
plt.title("VW Predictions (Porsche and BMW as Exog) vs Real Data", size=24)
plt.show()
## after plotting, we see that the new forecast is now similar to the one using only BMW prices
#  however this one matched the smaller shift better and adjusts to big jumps and drop faster, thus by adding a relevant value to the market benchmark we can once again improve predictions significantly 
## therefore we can confidently say that MAX models are less affected by real life events (exp new information) because the other time series often reflect the changes in the market and help navigate expectation  
## however we saw that the general trends changed after the first announcement even though it took some time for the prices to reasonably shift an unpredicted manner 

### Volatility ###
## Volatility of VW for Each Period 
df['sq_vol'][start_date:ann_1].plot(figsize=(20,5), color="#33B8FF")
df['sq_vol'][ann_1:ann_2].plot(color="#1E7EB2")
df['sq_vol'][ann_2:end_date].plot(color="#0E3A52")
plt.show()

## we see that Volkswagen has the highest volatility before any of the announcements, this is fascinating because the stocks becomes extremely stable following each the two announcements, while it exhibits instability in the time leading up to each purchases (rumors make uncertaintly)
## Garch(1,1) model should be the best fit for returns

model_garch_pre = arch_model(df.ret_vol[start_date:ann_1], mean="Constant", vol="GARCH", p=1, q=1)
results_garch_pre = model_garch_pre.fit(update_freq=5)

model_garch_btn = arch_model(df.ret_vol[ann_1:ann_2], mean="Constant", vol="GARCH", p=1, q=1)
results_garch_btn = model_garch_pre.fit(update_freq=5)

model_garch_post = arch_model(df.ret_vol[ann_2:end_date], mean="Constant", vol="GARCH", p=1, q=1)
results_garch_post = model_garch_pre.fit(update_freq=5)

print(results_garch_pre.summary())
## p value for Beta 1 (tidak signifikan), that meand the trends in variance are not as persistent as we would expect, so no need for garch components, and simple arch model would probably work better

print(results_garch_btn.summary())
## p value for beta 1 equal to zero (signifikan), this means the autocorrelation in the conditional variance is significant and hardly dies off based on the koeff value 
#  canstant mean is significant indicates there is some constant trend in the return values 
# simply put, we never expect returns or their volatility to ever be perfectly stable which is normal for a market lacking efficiency

print(results_garch_post.summary())
## omega, alpha 1 are non significant 

## there is different volatility trends in all 3 periods, koeff untuk variance juga tidak signifikan
##  VW buy porsche has resulted in much lower volatility after each announcement making VW stocks more appealing prospect 
