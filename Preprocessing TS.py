import numpy as np
import pandas as pd

raw_csv_data = pd.read_csv('D:/DATA ANALYST/belajar_python/TIME SERIES/Index2018.csv')
df_comp = raw_csv_data.copy()

### Mengelola Variabel "Date" Menjadi Data Jenis DateTime ###
print(df_comp.date.describe()) #top date is misleading karna date adalah string maka semua dihitung 1 dan diambil secara acak

df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True) #Data date diawali dengan Day maka "dayfirst= true"
print(df_comp.head())
print(df_comp.date.describe())

### Set Any Atrribute of a DataFrame as an Index ###
df_comp.set_index("date", inplace = True)
print(df_comp.head())
#print(df_comp.date.describe()) #kalo sudah jadi index, maka no longer save its values as separate attribute in DataFrame

### Setting the Desired Frequency (Time Series harus punya konstan frekuensi) ###
df_comp = df_comp.asfreq('b') #w= weekly, h= hour, d= day, a= annual, m=month, b= bussiness day
print(df_comp)

### Mengelola Missing Values ###
print(df_comp.isna())
print(df_comp.isna().sum()) #karena set frekuensi, menyebabkan data date nambah dengan missing values

df_comp.spx = df_comp.spx.fillna(method = "ffill")
print(df_comp.isna().sum())

df_comp.ftse = df_comp.ftse.fillna(method = "bfill")
print(df_comp.isna().sum())

df_comp.dax = df_comp.dax.fillna(value= df_comp.dax.mean())
print(df_comp.isna().sum())

df_comp.nikkei = df_comp.nikkei.fillna(method = "ffill")
print(df_comp.isna().sum())

### Adding and Removing Columns in a Data Frame ###
df_comp['market_value'] = df_comp.spx
print(df_comp.describe())

del df_comp['spx'], df_comp['ftse'], df_comp['dax'], df_comp['nikkei']
print(df_comp.describe())

### Split the Dataset ###
size = int(len(df_comp)* 0.8)
df = df_comp.iloc[:size]
df_test = df_comp.iloc[size:]

### Cek Overlapping Dataset ###
print(df.tail())
print(df_test.head()) #kalau data sama maka terjadi overlapping