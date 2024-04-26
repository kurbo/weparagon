# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:18:33 2024

@author: yangl
"""

import pandas as pd
import re

countyCode = '01003'
filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
incomefile = "income/" + countyCode + ".txt"
 
file_path = filepath + incomefile

# Define the header pattern using a regular expression
header_pattern = r'DATE\s+VALUE'

# Read the CSV file using read_csv with skiprows and header options
#df = pd.read_csv(file_path, sep='\t', skiprows=lambda x: not pd.read_csv(file_path, skiprows=x, header=None).iloc[0].str.contains(header_pattern).any())


# Determine the number of rows to skip until the header
#header_row = "DATE        VALUE"
skip_rows = 0
with open(file_path, 'r') as file:
#    skip_rows = sum(1 for line in file if line.strip() != header_row)
    for line in file:
        print(line)
        print("contains patter:", re.search(header_pattern, line))
        if re.search(header_pattern, line):
            break
        else:
            skip_rows += 1
            
        
    #skip_rows = sum(1 for line in file if not re.search(header_pattern, line))
    print('skip ', skip_rows, ' lines')

# Read the CSV file starting from the header row
df = pd.read_csv(file_path, skiprows=skip_rows, sep='\t')

# Display the resulting DataFrame
print(df.head(10))

import pandas as pd

# Assuming you have a DataFrame 'df' with a column 'DATE'
# Replace this with your actual DataFrame and column names
df = pd.DataFrame({
    'DATE': ['2013-01-01', '2013-01-02', '2013-01-03'],
    'Value': [10, 20, 15]
})

# Find the row where 'DATE' is '2013-01-01'
desired_date = '2013-01-01'
result_df = df[df['DATE'] == desired_date]

# Display the result
print(result_df)

import pandas as pd

# Assuming you have a DataFrame 'd1' with columns 'county_code', 'year', and other columns
# Replace this with your actual DataFrame and column names
d1 = pd.DataFrame({
    'county_code': ['01003', '01003', '01004', '01004'],
    'year': [2013, 2014, 2013, 2014],
    'other_column': [100, 200, 150, 250]
})

# Define values for the 'income' column
income_2013_value = 50000
income_2014_value = 55000

# Add a new column 'income' and set values based on conditions
d1['income'] = None  # Initialize the 'income' column with None

# Set values for the specific rows
d1.loc[(d1['county_code'] == '01003') & (d1['year'] == 2013), 'income'] = income_2013_value
d1.loc[(d1['county_code'] == '01003') & (d1['year'] == 2014), 'income'] = income_2014_value

# Display the resulting DataFrame 'd1'
print(d1)
