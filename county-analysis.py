# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:55:48 2024

@author: yangl
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
import re
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'

year =2021
dc = 500
file1='oom-county-'+ str(year) +'-FFT.txt'
df = read_csv(filepath+file1, sep='\t')
df["Year"] = year

# Data Manipulation
df_grouped = df.groupby(['County', 'Year'])['Deaths'].sum().reset_index()

# Identify Top Counties Each Year
#top_counties_each_year = df_grouped.loc[df_grouped.groupby('Year')['Deaths'].idxmax()]
top_counties_each_year = df_grouped.groupby('Year').apply(lambda group: group.nlargest(dc, 'Deaths')).reset_index(drop=True)

print(top_counties_each_year.head(dc))

# Calculate Total Deaths Each Year
total_deaths_each_year = df_grouped.groupby('Year')['Deaths'].sum().reset_index()

# Calculate Percentage of Deaths from Top Counties
top_counties_percentage = top_counties_each_year.merge(total_deaths_each_year, on='Year', suffixes=('_top_counties', '_total'))
top_counties_percentage['Percentage_Top_Counties'] = (top_counties_percentage['Deaths_top_counties'] / top_counties_percentage['Deaths_total']) * 100
print(top_counties_percentage['Percentage_Top_Counties'])
print(sum(top_counties_percentage['Percentage_Top_Counties']))

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar plot for Total Deaths
ax1.bar(total_deaths_each_year['Year'], total_deaths_each_year['Deaths'], label='Total Deaths', color='blue', alpha=0.7)

# Bar plot for Deaths from Top Counties
ax2 = ax1.twinx()
ax2.plot(top_counties_percentage['Year'], top_counties_percentage['Percentage_Top_Counties'], label='Percentage from Top Counties', color='red', marker='o')

# Customize the plot
ax1.set_title(f'Total and Percentage of Deaths from Top Counties ({year})')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Death Count', color='blue')
ax2.set_ylabel('Percentage from Top Counties', color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

# Plotting
plt.figure(figsize=(12, 8))
for year, data in top_counties_each_year.groupby('Year'):
    plt.bar(data['County'], data['Deaths'], label=str(year))

# Customize the plot
plt.title('Top 20 Counties with Highest Drug Overdose Deaths')
plt.xlabel('County')
plt.ylabel('Death Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Year')
plt.tight_layout()

# Show or save the plot
plt.show()

'''
import requests

for county_code in df['County Code']:
    cnty5 = str(county_code).zfill(5)
    filename = cnty5 + '.txt'
    url = 'https://fred.stlouisfed.org/data/PCPI'+ filename  # Replace with the actual URL

    print(url)
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the content of the response to a local file
        with open(filename, 'wb') as file:
            file.write(response.content)
        print('Saved ', filename)
    else:
        print(f'Failed to download the file. Status code: {response.status_code} ', filename)
'''
