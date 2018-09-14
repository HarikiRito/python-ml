# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Data_Refactor.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking care of missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X[:, 0] = LabelEncoder().fit_transform(X[:, 0]) # Transform List Of String To Number Index (Country)

X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray() # Categorize the Matrix at column 0
y = LabelEncoder().fit_transform(y)

# Split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test  = sc_x.transform(X_test)

print(y[0])
