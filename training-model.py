# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:05:18 2024

@author: Kirby Fung
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv

filepath = 'D:/Contests/MTFC/2023-24/Data/CSV/'
f1 = filepath + 'CDC-wonder/train-county-2013-2021.txt'
f2 = filepath + 'CDC-wonder/test-county-2022-2023.txt'

data = read_csv(f1)
data = data.dropna(subset=['meanincome'])

#data1 = data.drop(data.columns[0], axis=1) 

# save features as pandas dataframe for stepwise feature selection
X1 = data.drop(columns=['County', 'Deaths', 'Crude Rate'])
Y1 = data[['Deaths']].copy()
X1names = X1.columns

tst = read_csv(f2)
tst = tst.dropna(subset=['meanincome'])
tst_x = tst.drop(columns=['County', 'Deaths', 'Crude Rate'])
tst_y = tst[['Deaths']].copy()

print(X1.head(10))
print(Y1.head(10))

set_option('display.float_format', '{:.1f}'.format)
set_option('display.width', 600)
## fix display precision
set_option('display.precision', 1)
set_option('display.max_columns', None)
D1 = data.drop(columns=['Deaths', 'County Code', 'Year Code'])
description = D1.describe()
print("Descriptive Statistics of training data")
print(description)


# Determiniation of dominant features , Method one Recursive Model Elimination, 
# very similar idea to foreward selection but done recurssively. This method is gready
# which means it tries one feature at the time
# Start with largest feature numbers to observe the model score change
def RFE_loop(x, y, maxNum):
    print("Start with ", maxNum, "features, then trim down to see the score change")
    for i in range(maxNum, 0, -1):
        NUM_FEATURES = i 
        model = LinearRegression()

        ##fix the wrong "rfe = RFE(model, NUM_FEATURES)" unspecified argument issue by specifying the argument
        rfe = RFE(estimator=model, n_features_to_select=NUM_FEATURES)

        fit = rfe.fit(x, y)
        print("Num Features:", fit.n_features_)
        print("Selected Features:", fit.support_)
        print("Feature Ranking:", fit.ranking_)
        # calculate the score for the selected features
        score = rfe.score(X,Y)
        print("Model Score with selected features is: ", score)
        
X = X1.values
Y = Y1.values
RFE_loop(X, Y, len(X1names))

regression_model = LinearRegression()

X_train_selected = X1
y_train = Y1
regression_model.fit(X_train_selected, y_train)


# Make predictions on the test set
y_pred = regression_model.predict(tst_x)

# Evaluate the performance of the regression model
mse = mean_squared_error(tst_y, y_pred)
print(f"Linear Regression Model Mean Squared Error on Test Data: {mse}")

rmse = np.sqrt(mse)
print(f"Linear Regression Model Root Mean Squared Error on Test Data: {rmse}")

# hyperparameters can be adjusted further 
model_rf = RandomForestRegressor(n_estimators=100, random_state=42) 

# Train the model on the training data
model_rf.fit(X_train_selected, y_train)

y_pred = model_rf.predict(tst_x)

# Mean Squared Error
mse = mean_squared_error(tst_y, y_pred)
print(f"RF Model: MSE on Test Data: {mse}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"RF Model: RMSE on Test Data: {rmse}")


def display_score(scores) :
    print('Score:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())

#cross validate all the models used
def crosscheck(model, cvn, modelName, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=cvn, scoring='neg_mean_squared_error')

    # The scores are negative mean squared errors, as 'neg_mean_squared_error' is used
    # To get mean squared errors, take the negative of scores
    rmse_scores = np.sqrt(-scores)
    
    print(cvn, '-fold cross validation on model ', modelName, ' rmse scores on training data:')
    display_score(rmse_scores)

crosscheck(model_rf, 5, 'Random Forest', X_train_selected, y_train)
crosscheck(regression_model, 5, 'Multiple Linear Regression', X_train_selected, y_train)


X_train, X_val, y_train, y_val = train_test_split(X1, Y1)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
    
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
best_n_estimator = np.argmin(errors) + 1
    
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimator)
gbrt_best.fit(X_train, y_train)
    
yt_pred = gbrt_best.predict(tst_x)
mse = mean_squared_error(tst_y, yt_pred)
rmse = np.sqrt(mse)
print(f"GBRT-es Model: MSE on Test Data: {mse}")
print(f"GBRT-es Model: RMSE on Test Data: {rmse}")

crosscheck(gbrt_best, 5, 'GBRT-es', X_train, y_train)