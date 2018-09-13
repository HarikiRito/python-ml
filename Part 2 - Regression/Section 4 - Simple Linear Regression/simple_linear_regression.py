# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matrixC as mc
from sklearn.model_selection import train_test_split

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Feature Scalling

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test  = sc_x.transform(X_test)"""

# Fitting Linear Regression into Training Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predict the Test set
y_pred = regressor.predict(X_test)

Result = mc.flip([y_pred, y_test])

# Visualising Data
graph1 = graph2 = plt
graph1.scatter(X_train, y_train, color='red')
graph1.plot(X_train, regressor.predict(X_train), color='blue')
graph1.title('Salary vs Experience (Training set)')
graph1.xlabel('Years of Experience')
graph1.ylabel('Salary')
graph1.draw()
graph1.show()

graph2.scatter(X_test, y_test, color='red')
graph2.plot(X_train, regressor.predict(X_train), color='blue')
graph2.title('Salary vs Experience (Test set)')
graph2.xlabel('Years of Experience')
graph2.ylabel('Salary')
graph2.draw()
graph2.show()

print()
