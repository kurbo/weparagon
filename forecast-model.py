# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:48:31 2024

@author: yangl
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.ticker import MaxNLocator

filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
f1 = filepath + 'CDC-wonder/train-county-2013-2021.txt'
f2 = filepath + 'CDC-wonder/test-county-2022-2023.txt'
wd = filepath + 'CDC-wonder/county-2013-2023-noincome.txt'

# Create an array of empty DataFrames
miti_num = 5 
county_forecasts_miti = [pd.DataFrame() for _ in range(miti_num)]
aggregate_miti = [pd.DataFrame() for _ in range(miti_num)]

forecast_periods = 10

'''
# The following is based on the old files with income
df1 = read_csv(f1)
#data = data.dropna(subset=['meanincome'])
#data1 = data.drop(data.columns[0], axis=1) 

# save features as pandas dataframe for stepwise feature selection
X1 = df1.drop(columns=['Crude Rate'])
X1names = X1.columns

df223 = read_csv(f2)
# Drop rows where 'year' is equal to 2023
df2 = df223[df223['Year Code'] != 2023]

df = pd.concat([df1, df2], ignore_index=True)
df = df.drop('Crude Rate', axis=1)
'''


# This is based on new file without income from 2013-2022 (drop 2023)
dfni = read_csv(wd)
dfni = dfni.drop('Crude Rate', axis=1)
df = dfni[dfni['Year Code'] != 2023]
df = df[df['Year Code']!=2022]
#df = df[df['Year Code']!=2021]

df.set_index('Year Code', inplace=True, drop=False)

set_option('display.float_format', '{:.1f}'.format)
set_option('display.width', 600)
## fix display precision
set_option('display.precision', 1)
set_option('display.max_columns', None)


print(df.head(50))


# Forecast function
def generate_county_forecasts(forecast_periods, data, mitigation_factor={}, worsen_factor={}):
    county_forecasts = {}

    for county, county_data in data.groupby('County Code'):
        model = SARIMAX(county_data['Deaths'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        
        # Generate forecast
        forecast = results.get_forecast(steps=forecast_periods)
        
        # Save county-level forecast
        county_forecasts[county] = forecast.predicted_mean.values * mitigation_factor.get(county, 1.0) * worsen_factor.get(county, 1.0)

    return county_forecasts

def aggregate_f(periods, countyforecasts):    
    # Aggregate forecasts for each year
    aggregate_forecast = np.zeros(periods)
    for county, forecast in countyforecasts.items():
        aggregate_forecast += forecast
    return aggregate_forecast

# Generate county-level forecasts
county_forecasts = generate_county_forecasts(forecast_periods, df)
print(county_forecasts)

aggregate_forecast = aggregate_f(forecast_periods, county_forecasts)


# generate mitigation forecasts
df_sorted = df.sort_values(by='Deaths', ascending=False)
def mitival(data, mfactor, n, defm=1.0):
    miti = {}
    i = 0
    for county, county_data in data.groupby('County Code'):
        if i<n:
            miti[county] = mfactor
        else:
            miti[county] = defm
    return miti

idx = 0
for mfactor in [0.9603424658, 0.9439, 0.9216837983, 0.9427891899] :
    miti = mitival(df, mfactor, 9999)
    
    #print("mitigation array")
    #print(miti)
    county_forecasts_miti[idx] = generate_county_forecasts(forecast_periods, df, miti)
    aggregate_miti[idx] = aggregate_f(forecast_periods, county_forecasts_miti[idx])
    idx += 1

mfactor = 0.8433675967
miti = mitival(df, mfactor, 400, mfactor/2)
county_forecasts_miti[idx] = generate_county_forecasts(forecast_periods, df, miti)
aggregate_miti[idx] = aggregate_f(forecast_periods, county_forecasts_miti[idx])
 

# Plot the original data and aggregated forecast
plt.figure(figsize=(12, 6))

# Reset the index to make 'year' a regular column
#df = df.reset_index()
# show original aggregated death data
# Group by year and sum the deaths
# Group by the 'year' column without resetting the index
year_df = df.groupby(df.index.get_level_values('Year Code'))['Deaths'].sum()
#year_df = df.groupby('Year Code')['Deaths'].sum()

print("existing deaths total:")
print(year_df)

# Plot the aggregated deaths per year
plt.plot(year_df, label='Opioid Overdoes Deaths from 2013-2022')

#plt.bar(year_df.index, year_df, width=0.7, label='Opioid Overdoes Deaths from 2013-2022')


#plt.bar(df.index, df['Deaths'], width=0.7, label='Original Data')
'''
#plot the actual 2023 deaths data 
df223 = read_csv(wd)
dy223 = df223[df223['Year Code'] == 2023]
dy223.set_index('Year Code', inplace=True)
y223=dy223.groupby('Year Code')['Deaths'].sum()
plt.bar(y223.index, y223, width=0.7)
'''

# Aggregated Forecast
forecast_years = np.arange(df['Year Code'].max() + 1, df['Year Code'].max() + 1 + 10)
#plt.plot(forecast_years, aggregate_forecast, linestyle='--', label='opioid Overdose Deaths Forecast (2023-2032)', color='yellow')

'''
forecast_bars = plt.bar(forecast_years.astype(int), aggregate_forecast, width=0.7, alpha=0.7, label='opioid Overdose Deaths Forecast (2023-2032)', edgecolor='black', color='yellow')
for bar in forecast_bars:
    bar.set_linestyle('--')  # Dotted line style
    bar.set_linewidth(2)     # Line width
'''
    
# Set the x-axis locator to integer values
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #forecast_bars_miti = plt.bar(forecast_years, am, width=0.7, alpha=0.7, label='Aggregated Forecast', edgecolor='black')
    #for bar in forecast_bars_miti:
    #    bar.set_linestyle('--')  # Dotted line style
    #    bar.set_linewidth(2) 

#plot continuous lines
forecastYrs = np.arange(df['Year Code'].min(), df['Year Code'].max() + 1 + 10)
af = np.concatenate((year_df.values, aggregate_forecast))
plt.plot(forecastYrs, af, linestyle='--', label='Opioid Overdose Deaths Forecast (2023-2032)', color='yellow')
Fsum = np.sum(af)

## describe the growth rate in terms of frequency
# Calculate the yearly growth rate using the percentage change
afs = pd.Series(af)
pcf = afs.pct_change() * 100
# Display summary statistics of the growth rate
with open(filepath+'model-log.txt', 'w') as fil:   
    print("statistics of the growth rate on the forecast", file=fil)
    print(pcf.describe(), file=fil)
    print("all forecast deaths from 2013 to 2032", file=fil)
    print(af, file=fil)

'''
mititle = ["Forecast with Law Enforcement Mitigation", "Forecast with DUA Treatment Mitigation", "Forecast with Prevention Mitigation (OEND)", "Forecast with Insurance Mitigation", "Forecast with Enhanced County Mitigation (K=400, N=2)" ]
idx = 0
Dsum = np.zeros(miti_num)
Dif = np.zeros(miti_num)
for am in aggregate_miti:
    af = np.concatenate((year_df.values, am))
    plt.plot(forecastYrs, af, linestyle='--', label=mititle[idx])
    Dsum[idx] = np.sum(af)
    Dif[idx] = Fsum - Dsum[idx]
    idx += 1

print("Lives saved by mitigations")
print(Dif)
'''

# Plot settings
plt.title('Opioid Overdose Deaths Forecast with Mitigations in 2023-2032')
plt.xlabel('Year')
plt.ylabel('Total Deaths in US')
plt.legend()
plt.show()




# display heatmap after forecast and mitigation
## come back when have time
"""
file2 = filepath + 'uscounties/uscounties.csv' #county longtitude/latitude data

def plot_county_oom(df1, year, title, countyCode='County Code', county='County'):
    df2 = read_csv(file2)

    df1['deathrate'] = (df1['Deaths'] / df1['Population']) * 100000
    # Perform the merge using the 'id' and 'county_fips' fields
    merged_df = pd.merge(df1, df2, left_on=countyCode, right_on='county_fips', how='inner')

    merged_df.plot(kind='scatter', x='lng', y='lat', alpha=1, s=merged_df["Deaths"]/10, 
                   label="Death", figsize=(10,7), c="Deaths", cmap=plt.get_cmap("jet"), 
                   colorbar=True)
   
    ''' 
    # this is for deathrate
    merged_df.plot(kind='scatter', x='lng', y='lat', alpha=1, s=merged_df["deathrate"]/2, 
                  label="Death Rate", figsize=(10,7), c="deathrate", cmap=plt.get_cmap("jet"), 
                  colorbar=True)
    '''
    
    plt.legend()
    #plt.title("Opioid Overdose Deaths Forecast by County in " + str(year))
    plt.title(title)
    plt.show()    
"""

