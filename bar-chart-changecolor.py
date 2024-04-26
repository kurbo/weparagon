# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:08:31 2024

@author: yangl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Normalize the values to [0, 1] to use in colormap
normalized_values = np.array(values) / max(values)

# Choose a colormap (e.g., 'viridis' for a perceptually uniform colormap)
colormap = plt.cm.viridis

# Generate a bar chart with colors based on the colormap
plt.bar(categories, values, width=1, color=colormap(values), edgecolor='black')

# Add labels and title
plt.title('Bar Chart with Darker Colors for Higher Numbers')
plt.xlabel('Categories')
plt.ylabel('Values')

# Show the plot
plt.show()

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Choose the 'Blues' colormap
colormap = plt.cm.Blues

# Generate a bar chart with colors based on the colormap
plt.bar(categories, values, width=1, color=colormap(values), edgecolor='black')

# Add labels and title
plt.title('Bar Chart with Blues Colormap')
plt.xlabel('Categories')
plt.ylabel('Values')

# Show the plot
plt.show()

categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Choose the 'viridis' colormap
viridis = plt.cm.get_cmap('viridis')

# Truncate the Viridis colormap to emphasize blue
blue_viridis = LinearSegmentedColormap.from_list('blue_viridis', viridis(np.linspace(0.2, 1, 256)))

# Generate a bar chart with colors based on the truncated colormap
plt.bar(categories, values, color=blue_viridis(values), edgecolor='black')

# Add labels and title
plt.title('Bar Chart with Truncated Viridis Colormap (Emphasizing Blue)')
plt.xlabel('Categories')
plt.ylabel('Values')

# Show the plot
plt.show()

# Choose the 'viridis' colormap
viridis = plt.cm.get_cmap('viridis')

# Reverse the Viridis colormap
reversed_viridis = LinearSegmentedColormap.from_list('reversed_viridis', viridis(np.linspace(0.5, 1, 256)))

# Generate a bar chart with colors based on the reversed colormap
plt.bar(categories, values, width=1, color=reversed_viridis(values), edgecolor='black')

# Add labels and title
plt.title('Bar Chart with Reversed Viridis Colormap (Darker Blue for Higher Numbers)')
plt.xlabel('Categories')
plt.ylabel('Values')

# Show the plot
plt.show()
