# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:37:31 2024

@author: Kirby Fung
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
file2 = 'uscounties/uscounties.csv' #county longtitude/latitude data

def plot_county_oom(year, countyCode, county):
    file1='oom-county-'+ str(year) +'-FFT.txt'
    df1 = read_csv(filepath+file1, sep='\t')
    df2 = read_csv(filepath+file2)

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
    plt.title("Opioid Overdose Death Count by County in " + str(year))
    
    # Assuming 'merged_df' is your DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df_sort = merged_df.sort_values(by='Deaths', ascending=False)
    #df_sort = merged_df.sort_values(by='deathrate', ascending=False)
    
    selected_cols = [county, 'deathrate', 'Deaths', 'Population']
    print("County Ranking in ", year)
    print(df_sort[selected_cols].head(15).to_string(index=False))

#plot year 2018-2020 county-level opioid-overdose-mortality data
#for year in range(2015, 2021):
    #plot_county_oom(year, 'County Code', 'County')
    
plot_county_oom(2013, 'County Code', 'County')

for year in range(2021, 2024):
    plot_county_oom(year, 'Residence County Code', 'Residence County')

   
def county_crime(year, num):
    file1 = 'cr/' + str(year) + '-cr.txt'
    df1 = read_csv(filepath+file1, sep='\t')

    selected_col = ['FIPS_ST', 'FIPS_CTY', 'CPOPARST', 'GRNDTOT', 'DRUGTOT']
    print(df1[selected_col].head(num))


county_crime(2013, 200)