# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:26:14 2024

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

xcol = "Year"
ycol = "Deaths"

filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
filename = 'oom-total-1999-2020-FFT.txt'

df = read_csv(filepath+filename, sep='\t')

#groupby to sum up all 50 states death data based on age groups
result = df.groupby(xcol)[ycol].sum().reset_index()
print(result)

result.plot.bar(x=xcol, y=ycol, edgecolor='black')
plt.title('Opioid Overdose Deaths from 1999 to 2020')
#plt.ylabel('Number of Deaths')
plt.show()
