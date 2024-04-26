# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:52:31 2024

@author: yangl
"""

import matplotlib.pyplot as plt

# Sample data
labels = ['Category A', 'Category B', 'Category C']
values = [30, 70, 50]

# Find the index of the category with the largest value
largest_index = values.index(max(values))

# Generate a list of colors with a custom color for the largest portion and None for the rest
colors = ['lightblue' if i == largest_index else None for i in range(len(labels))]

# Generate a pie chart with custom color for the largest portion
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

# Add a legend
plt.legend(labels, title='Category Names', loc='center left', bbox_to_anchor=(1, 0.5))

# Add a title
plt.title('Pie Chart with Custom Color for Largest Portion')

# Show the plot
plt.show()