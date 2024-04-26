# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:36:17 2024

@author: yangl
"""

import pandas as pd

# Assuming you have a DataFrame 'd1' with some columns having ".0" values
# Replace this with your actual DataFrame
d1 = pd.DataFrame({
    'Column1': [1.0, 2.0, 350],
    'Column2': [4.0, 5.0, 6.0],
    'Column3': ['7.0', '8.0', '9.0']
})

# Copy the DataFrame to avoid modifying the original one
d2 = d1.copy()

# Iterate through columns and convert values to integers
for column in d2.columns:
    d2[column] = d2[column].astype(str).str.rstrip('.0').astype(int)

# Display the resulting DataFrame 'd2'
print(d2)

import pandas as pd

# Assuming you have a DataFrame 'df' with a column 'c'
# Replace this with your actual DataFrame
df = pd.DataFrame({
    'c': ["4.2", "1.0", "2.0", "3.7"]
})

# Create a new DataFrame 'df_new' based on the condition
df_new = df.copy()
df_new['c'] = df_new['c'].apply(lambda x: int(float(x)) if float(x) % 1 == 0 else x)

# Display the resulting DataFrame 'df_new'
print(df_new)


import pandas as pd

# Assuming you have a DataFrame 'd1' with an integer column 'integer_column'
# Replace this with your actual DataFrame and column name
d1 = pd.DataFrame({
    'integer_column': [1, 23, 456, 7890]
})

# Create a new DataFrame 'd2' with the formatted column
d2 = pd.DataFrame({
    'formatted_column': d1['integer_column'].astype(str).str.zfill(5)
})

# Display the resulting DataFrame 'd2'
print(d2)