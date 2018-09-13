# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

df = pd.DataFrame(data = X, columns = ['Country', 'Age' , 'Salary', 'Purcharsed'])

df.to_csv('Data_Refactor.csv', index = False)
