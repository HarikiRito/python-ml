# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:].values
y = dataset.iloc[:, 3].values


# Split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scalling

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test  = sc_x.transform(X_test)"""