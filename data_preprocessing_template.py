# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 18:41:52 2016

This is a more complete preprocessing teplate, which considers missing values,
categorical data encoding and feature scaling.

@author: Shervin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# Encoding categorized data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
countryEncoder = LabelEncoder()
hotEncoder = OneHotEncoder(categorical_features = [0])
x[:,0] = countryEncoder.fit_transform(x[:,0])
x = hotEncoder.fit_transform(x).toarray()

# Encoding the dependent variable
purchasedEncoder = LabelEncoder()
y = purchasedEncoder.fit_transform(y)

# Splitting dataset
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
xTrain = xScaler.fit_transform(xTrain)
xTest = xScaler.transform(xTest)
