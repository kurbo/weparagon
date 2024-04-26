# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:14:22 2024

@author: Kirby Fung
"""

# Function to filter rows based on the pattern
'''
def filter_rows(row):
    value = row["Five-Year Age Groups Code"]
    if isinstance(value, (int, float)):
        # Keep the value if it's a float
        return value
    elif isinstance(value, str):
        # Check if the string matches a pattern
        match = re.match(r'^(\d+)-', value)
        return match.group(1) if match else None
    else:
        return None
'''

import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
import re
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

bar_width=1
xcol = 'Five-Year Age Groups Code'
ycol = 'Deaths'
xcol_new = 'agecode'

def filter_row(row):
        value = row[xcol];
        if value == 'NS': 
            return None
        else:
            return value
    
# Define custom sorting order
custom_order = {'1':0, '1-4': 1, '5-9': 2, '10-14': 3, '15-19': 4, '20-24': 5, '25-29':6, 
                '30-34':7, '35-39':8, '40-44':9, '45-49':10, '50-54':11, '55-59':12,
                '60-64':13, '65-69':14, '70-74':15, '75-79':16, '80-84':17, '85-89':18,
                '90-94':19, '95-99':20, '100+':21}

def custom_sort_key(col):
    return custom_order[col]
    
filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
#filename = 'oom-2020.txt'
filename = 'oom-5yr-age-2022.txt'

df = read_csv(filepath+filename, sep='\t')


# Apply the filter function to create a new column 'Number'
df[xcol_new] = df.apply(filter_row, axis=1)

cols_keep = [xcol_new, ycol]
opdf = df.dropna(subset=[xcol_new])
opdf1= opdf[cols_keep]

set_option('display.width', 100)
set_option('display.max_columns', None)

# generate the histogram of the distribution of data and its descriptive statistics
#opdf1.T.hist()

#groupby to sum up all 50 states death data based on age groups
result = opdf1.groupby(xcol_new)[ycol].sum().reset_index()

# Sort the DataFrame based on the custom sorting key
result['SortingKey'] = result[xcol_new].apply(custom_sort_key)
df_result = result.sort_values(by='SortingKey').drop('SortingKey', axis=1)
print(df_result)

#make the color changing based on numbers
normalized_values = np.array(df_result[ycol]) / max(df_result[ycol])

viridis = plt.cm.get_cmap('viridis')

# Truncate the Viridis colormap to emphasize blue
blue_viridis = LinearSegmentedColormap.from_list('blue_viridis', viridis(np.linspace(0.3, 0.1, 256)))

df_result.plot.bar(x=xcol_new, y=ycol, width=bar_width, color=blue_viridis(normalized_values), edgecolor='black')

# Remove the legend
plt.legend([])
plt.title('Opioid Overdose Deaths Distribution in 2022')
plt.xlabel('Age group')
plt.ylabel('Number of Deaths')
plt.show()


'''
Process race-based analysis 
'''
file_race = 'oom-race-state-2015-2020-F11-F19-T40.txt'
xcol = 'Race Code'
xcol2 = 'Race'
df = read_csv(filepath+file_race, sep='\t')


cols_keep = [xcol2, xcol, ycol]
opdf= df[cols_keep]

set_option('display.width', 100)
set_option('display.max_columns', None)

#groupby to sum up all 50 states death data based on age groups
result = opdf.groupby(xcol)[ycol].sum().reset_index()
print(result)
result.plot.pie(y=ycol, labels=None, autopct='%1.1f%%', startangle=90)
plt.legend(result[xcol], bbox_to_anchor=(1, 0.5), loc='center left')
plt.title('Opioid Overdose Deaths Race Distribution')
# Remove the y-axis label
plt.ylabel('')
plt.show()

file_race = 'oom-race-state-2015-2020-F11-F19-T40.txt'
xcol = 'Race Code'
xcol2 = 'Race'
df = read_csv(filepath+file_race, sep='\t')


cols_keep = [xcol2, xcol, ycol]
opdf= df[cols_keep]

set_option('display.width', 100)
set_option('display.max_columns', None)

#groupby to sum up all 50 states death data based on age groups
result = opdf.groupby(xcol)[ycol].sum().reset_index()
print(result)
result.plot.pie(y=ycol, labels=None, autopct='%1.1f%%', startangle=90)
plt.legend(result[xcol], bbox_to_anchor=(1, 0.5), loc='center left')
plt.title('Opioid Overdose Deaths Race Distribution')
# Remove the y-axis label
plt.ylabel('')
plt.show()

'''
Analyze Race-Hispanic-origin-based data
'''
file_race = 'oom-race-hispanic-state-2015-2020-FFT.txt'
xcol = 'Race Code'
xcol2 = 'Race'
xcol3 = 'Hispanic Origin Code'
labels = ['American Indian or Alaska Native Hispanic ', 'Black or African American Hispanic ',
          'White Hispanic', 'Asian or Pacific Islander Hispanic', 
          'American Indian or Alaska Native Non-Hispanic', 
          'Black or African American Non-Hispanic',
          'White Non-Hispanic', 'Asian or Pacific Islander Non-Hispanic']

df = read_csv(filepath+file_race, sep='\t')


# Apply the filter function to filter out 'NS' hispanic code
# Sample filter function
def filter_function(row, column_name):
    # Your filtering logic here
    return row[column_name] != 'NS'

# Apply the filter function along rows
filtered_df = df[df.apply(lambda row: filter_function(row, xcol3), axis=1)]

cols_keep = [xcol3, xcol2, xcol, ycol]
opdf= filtered_df[cols_keep]

set_option('display.width', 100)
set_option('display.max_columns', None)

#groupby to sum up all 50 states death data based on age groups
result = opdf.groupby([xcol3, xcol])[ycol].sum().reset_index()
print(result)

# Find the index of the category with the largest value
largest_index = result[ycol].idxmax()
print('largest index ' + str(largest_index))

# Generate a list of colors with a custom color for the largest portion and None for the rest
custom_colors = ['brown', 'lightblue', 'red', 'lightyellow', 'lightcoral', 'gold', 'maroon', 'yellow']
#colors = ['lightblue' if i == largest_index else None for i in range(len(labels))]
explode = (0, 0, 0, 0, 0, 0, 0, 0)
def my_autopct(pct):
    return f'{pct:1.1f}%' if pct >= 1 else ''

result.plot.pie(y=ycol, labels=None, autopct=my_autopct, startangle=90, colors=custom_colors, explode=explode, shadow=True)
plt.legend(labels, bbox_to_anchor=(1, 0.5), loc='center left')
plt.title('Ethnicity Distribution on Opioid Overdose Deaths (2015-2020)', color='black')
# Remove the y-axis label
plt.ylabel('')
# Get the current figure
fig = plt.gcf()

# Set the background color of the figure
fig.set_facecolor('lightblue')
plt.show()


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the pie chart in 3D using DataFrame data
ax.pie(result[ycol], labels=None, autopct='%1.1f%%', startangle=90)
ax.title("pie chart")
#plt.legend(labels, bbox_to_anchor=(1, 0.5), loc='center left')
#plt.title('Opioid Overdose Deaths Hispanic-Origin Race Distribution')
# Remove the y-axis label
#plt.ylabel('')
plt.show()
'''
