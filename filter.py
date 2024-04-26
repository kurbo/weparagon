# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:13:47 2024

@author: yangl
"""

import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

df = pd.DataFrame(data)

# Sample filter function
def filter_function(row, column_name):
    # Your filtering logic here
    return row[column_name] > 4

# Apply the filter function along rows
filtered_df = df[df.apply(lambda row: filter_function(row, 'B'), axis=1)]

# Display the result
print(filtered_df)