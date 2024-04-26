# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:59:11 2024

@author: yangl
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
import re

def processfile(input_file, output_file_total, output_file_clean, pattern):
    # Open the input file for reading and the output files for writing
    with open(input_file, 'r') as infile, open(output_file_total, 'w') as outfile_total, open(output_file_clean, 'w') as outfile_clean:
        # Iterate through each line in the input file
        for line in infile:
            # Check if the line contains the pattern "total"
            if pattern in line.lower():
                # If it does, write it to the output file for totals
                outfile_total.write(line)
            else:
                # If it doesn't, write it to the output file for clean lines
                outfile_clean.write(line)
 
filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
inputf = 'opioid-overdose-mortality-2020.txt'
outputT = 'oom-total-2020.txt'
outputf = 'oom-2020.txt'

processfile(filepath+inputf, filepath+outputT, filepath+outputf, 'total')

#preprocess the age group into 1, 5, 10, ...
data = {'Pattern': ['123-abc', 'def-456', '789-ghi', 'jkl-mno']}
df = pd.DataFrame(data)

# Function to filter rows based on the pattern
def filter_rows(row):
    match = re.match(r'^(\d+)-', row['Pattern'])
    return match.group(1) if match else None

# Apply the filter function to create a new column 'Number'
df['Number'] = df.apply(filter_rows, axis=1)

# Drop rows where 'Number' is None
df = df.dropna(subset=['Number'])

# Display the resulting DataFrame
print(df)