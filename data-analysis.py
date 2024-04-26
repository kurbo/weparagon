# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:46:40 2024

@author: Kirby Fung
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

filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
f1 = filepath + 'CDC-wonder/train-county-2013-2021.txt'
f2 = filepath + 'CDC-wonder/test-county-2022-2023.txt'

d1 = read_csv(f1)
d2 = read_csv(f2)
df = pd.concat([d1, d2], ignore_index=True)

# Calculate deathrate and add it as a new column
df['deathrate'] = df['Deaths'] / df['Population']

def calpercent(selected_year, col, num):
    # Filter DataFrame for the selected year
    df_selected_year = df[df['Year Code'] == selected_year]

    # Get the top num counties with the highest deathrate
    top_counties = df_selected_year.nlargest(num, col)
    
    # Calculate the sum of deaths in the top 20 counties
    sum_deaths_top_counties = top_counties['Deaths'].sum()

    # Calculate the total deaths in the selected year
    total_deaths_selected_year = df_selected_year['Deaths'].sum()

    # Calculate the percentage
    percentage = (sum_deaths_top_counties / total_deaths_selected_year) * 100
    return {'percent' : percentage, 'year': selected_year}

'''    
    print(f"Top {num} counties with highest deathrate in {selected_year}:\n{top_counties}")
    print(f"\nSum of deaths in the top {num} counties: {sum_deaths_top_counties}")
    print(f"Total deaths in {selected_year}: {total_deaths_selected_year}")
    print(f"Percentage of deaths in the top 20 counties: {percentage:.2f}%")
'''

df1 = pd.DataFrame(columns=['percent', 'year'])
df2 = pd.DataFrame(columns=['percent', 'year'])
num_county = 400
for i in range(2013, 2023):
    dr_row = calpercent(i, 'deathrate', num_county);
    new_df = pd.DataFrame([dr_row])

    # Concatenate the original DataFrame with the new DataFrame
    df1 = pd.concat([df1, new_df], ignore_index=True)

    #df1 = df1.append(dr_row, ignore_index=True)

    d_row = calpercent(i, 'Deaths', num_county)
    new_df = pd.DataFrame([d_row])
    df2 = pd.concat([df2, new_df], ignore_index=True)

print(df1)
print(df2)

df1['year'] = df1['year'].astype(int)
df2['year'] = df2['year'].astype(int)

# Plot both DataFrames on the same graph
plt.plot(df2['year'], df2['percent'], label='Deaths in top 400 counties ranked by Death Count / total deaths', marker='o')
plt.plot(df1['year'], df1['percent'], label='Deaths in top 400 counties ranked by Death Rate / total deaths', marker='o')

# Ensure all years are displayed on the x-axis
plt.xticks(df1['year'])

# Add labels and a legend
plt.xlabel('Year')
plt.ylabel('Percent (%)')
plt.legend()

# Show the plot
plt.show()