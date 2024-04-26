# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:01:08 2024

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
file1 = 'CDC-wonder/oom-county-1999-2020.txt'
file2 = 'CDC-wonder/oom-county-2018-2024-p.txt'
destfile = 'CDC-wonder/train-new-oom-county-2013-2021.txt'
destfile2 = 'CDC-wonder/test-new-oom-county-2022-2023.txt'
dest1 = 'CDC-wonder/train-county-2013-2021.txt'
dest2 = 'CDC-wonder/test-county-2022-2023.txt'

wd = 'CDC-wonder/county-2013-2023-noincome.txt'

errlog = filepath + 'error_log.txt'
yearCode = 'Year Code'

df1 = read_csv(filepath+file1, sep='\t')
df2 = read_csv(filepath+file2, sep='\t')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
## fix display precision
pd.set_option('display.precision', 0)


# Filter rows based on the condition (year in [2013, 2020])
condition = (df1[yearCode] >= 2013) & (df1[yearCode] <= 2020)
d1 = df1[condition]
print(d1.head(10))

'''
This section is to trim off the extra .0 for precision=1, since Mariah generated the txt using default precision=1
not necessary for other files as I set precision=0 when I am downloading the data from CDC WONDER
'''
columns = ["County", "County Code", "Year Code", "Deaths", "Population", "Crude Rate"]
dc1 = d1[columns]
dc1_new = dc1.copy()
print(dc1.head(10))
for cl in ["County Code", "Year Code", "Deaths", "Population"]:
    dc1_new[cl] = dc1_new[cl].apply(lambda x: int(float(x)) if float(x) % 1 == 0 else x)

#format county code to 5 digits
dc1_new['County Code'] = dc1_new['County Code'].astype(str).str.zfill(5)
dc1_new.to_csv(filepath+destfile, index=False)


condition2 = (df2[yearCode] == 2021)
d2 = df2[condition2]
print(d2.head(10))

columns = ["Residence County", "Residence County Code", "Year Code", "Deaths", "Population", "Crude Rate"]
dc2 = d2[columns].rename(columns={
    'Residence County': 'County',
    'Residence County Code': 'County Code'
})
print(dc2.head(10))

#format county code to 5 digits
dc2['County Code'] = dc2['County Code'].astype(str).str.zfill(5)
dc2.to_csv(filepath+destfile, mode='a', header=False, index=False)

'''
Next, to add avg. income data into the training county data file
'''
#append the 2021 dataframe to 2013-2020 dataframe
df_1320 = pd.concat([dc1_new, dc2], ignore_index=True)
#df_1320 = dc1_new.append(dc2, ignore_index=True)


# Select rows where 'yearCode' is 2021, 2022, or 2023
selected_rows = df2[df2['Year Code'].isin([2021, 2022, 2023])]

dc2_new = selected_rows[columns].rename(columns={
    'Residence County': 'County',
    'Residence County Code': 'County Code'
})
print(dc2_new.head(10))
dfall = pd.concat([dc1_new, dc2_new], ignore_index=True)
dfall.to_csv(filepath+wd, index=False) 

unique_county_codes = df_1320['County Code'].unique()
import requests


'''    
for county_code in unique_county_codes:
    cnty5 = str(county_code).zfill(5)
    fname = cnty5 + '.txt'
    filename = filepath + 'income/' + fname
    url = 'https://fred.stlouisfed.org/data/PCPI'+ fname  # Replace with the actual URL

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

'''
# Add a new column 'income' and set values based on conditions
df_1320['meanincome'] = None  # Initialize the 'income' column with None

idx = 0
for countyCode in unique_county_codes:
    
    incomefile = "income/" + countyCode + ".txt"
    
    
    # Define the header pattern using a regular expression
    header_pattern = r'DATE\s+VALUE'

    # Determine the number of rows to skip until the header
    #header_row = "DATE        VALUE"
    skip_rows = 0
    file_path = filepath + incomefile
    try:
        with open(file_path, 'r') as file:
            #    skip_rows = sum(1 for line in file if line.strip() != header_row)
            for line in file:
                #print(line)
                #print("contains patter:", re.search(header_pattern, line))
                if re.search(header_pattern, line):
                    break
                else:
                    skip_rows += 1
                            
        #skip_rows = sum(1 for line in file if not re.search(header_pattern, line))
        print('skip ', skip_rows, ' lines')

        # Read the CSV file starting from the header row
        dincome = read_csv(file_path, skiprows=skip_rows, sep='\t*\s+\t*')
        dincome.columns = dincome.columns.str.strip()    
        print(dincome.head(2))
      
        #ccc is no use
        #for ccc in dincome.columns:
         #   print(ccc)
            
            for yr in range(2013, 2022):
                icol = str(yr) + '-01-01'
                print("find year:", icol)
                # Find the value in d2 where DATE is '2013'
                
                ic = dincome[dincome['DATE'] == icol]

                if ic.empty:
                    print("county ", countyCode, ' does not have income data for year ', yr )
                else:
                    icvalue = dincome.loc[dincome['DATE'] == icol, 'VALUE'].iloc[0]
                    # Set values for the specific rows
                    condition = (df_1320['County Code'] == countyCode) & (df_1320['Year Code'] == yr)
                    print('county ', countyCode, ', Year=', yr, ' income=', ic['VALUE'], ' row:',  df_1320[condition])
                    df_1320.loc[condition, 'meanincome'] = icvalue  
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")

        # Redirect the error message to an error log file
        with open(errlog, 'a') as error_file:
           print(f"Error: The file '{file_path}' does not exist.", file=error_file)
        

df_1320.to_csv(filepath+dest1, index=False)    
'''

'''
Following is to create the test data file including oom data from [2022,2023]
'''

'''
condition3 = (df2[yearCode] >2021)
d3 = df2[condition3]
print(d3.head(10))

columns = ["Residence County", "Residence County Code", "Year Code", "Deaths", "Population", "Crude Rate"]
dc3 = d3[columns].rename(columns={
    'Residence County': 'County',
    'Residence County Code': 'County Code'
})
print(dc3.head(10))

#format county code to 5 digits
dc3['County Code'] = dc3['County Code'].astype(str).str.zfill(5)
#dc3.to_csv(filepath+destfile2, index=False)


dc3["meanincome"] = None


def addIncome(df, unique_county_code):

    for countyCode in unique_county_codes:   
        incomefile = "income/" + countyCode + ".txt"
       # print("open county income file:", incomefile)
    
        # Define the header pattern using a regular expression
        header_pattern = r'DATE\s+VALUE'

        # Determine the number of rows to skip until the header
        #header_row = "DATE        VALUE"
        skip_rows = 0
        file_path = filepath + incomefile
        print("pre-open file:", file_path)
        try:
            with open(file_path, 'r') as file:
                print("open county income file:", incomefile)
                #    skip_rows = sum(1 for line in file if line.strip() != header_row)
                for line in file:
                    #print(line)
                    #print("contains patter:", re.search(header_pattern, line))
                    if re.search(header_pattern, line):
                        break
                    else:
                        skip_rows += 1
                            
                #skip_rows = sum(1 for line in file if not re.search(header_pattern, line))
                print('skip ', skip_rows, ' lines')

            # Read the CSV file starting from the header row
            dincome = read_csv(file_path, skiprows=skip_rows, sep='\t*\s+\t*')
            dincome.columns = dincome.columns.str.strip()    
            print(dincome.head(2))
       
            for yr in range(2022, 2023):
                print('for loop year=', yr)
                icol = str(yr) + '-01-01'
                print("find year:", icol)
                # Find the value in d2 where DATE is '2013'
                
                ic = dincome[dincome['DATE'] == icol]

                if ic.empty:
                   print("county ", countyCode, ' does not have income data for year ', yr )
                else:
                    icvalue = dincome.loc[dincome['DATE'] == icol, 'VALUE'].iloc[0]
                    # Set values for the specific rows
                    condition = (df['County Code'] == countyCode) & (df['Year Code'] == yr)
                    df.loc[condition, 'meanincome'] = icvalue
                    print('county ', countyCode, ', Year=', yr, ' income=', icvalue, ' row:',  df[condition])
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' does not exist.")

            # Redirect the error message to an error log file
            with open(errlog, 'a') as error_file:
                print(f"Error: The file '{file_path}' does not exist.", file=error_file)
                 
addIncome(dc3, unique_county_codes)

print(dc3.head(2))
dc3.to_csv(filepath+dest2, index=False)
'''