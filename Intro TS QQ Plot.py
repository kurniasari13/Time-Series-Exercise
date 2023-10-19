import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

raw_csv_data = pd.read_csv('D:/DATA ANALYST/belajar_python/TIME SERIES/Index2018.csv')
df_comp = raw_csv_data.copy()
print(df_comp.head())
print(df_comp)
print(df_comp.describe())

### Mencari Missing Value ###

print(df_comp.isna()) #Cek per row
print(df_comp.isna().sum()) #nilai 1 is True (True= ada missning values), jika jumlah berbeda maka ada missing values
print(df_comp.spx.isna().sum()) #cek missing value per kolom, "spx" is column name

### Plotting the Data ###

#US
df_comp.spx.plot(figsize=(20,5), title="S&P500 Prices")
plt.show()
#UK
df_comp.ftse.plot(figsize=(20,5), title="FTSE100 Prices")
plt.show()
#US & UK
df_comp.spx.plot(figsize=(20,5), title="S&P500 Prices")
df_comp.ftse.plot(figsize=(20,5), title="FTSE100 Prices")
plt.title("S&P vs FTSE") #ada misleading dan belum bisa dibanidngkan karena ada different magnitude
plt.show()

### QQ Plot ###
import scipy.stats
import pylab
#y axis is prices dr lower ke high, x axis 

scipy.stats.probplot(df_comp.spx, plot=pylab)
pylab.show()