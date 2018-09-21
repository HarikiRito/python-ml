# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matrixC as mc
from sklearn.model_selection import train_test_split

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Categorize Dataset
X = mc.categorical(X, 3)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# Split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=0)

# Feature Scalling

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test  = sc_x.transform(X_test)"""

# Fitting Multiple Linear Regression into Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

Result = mc.flip([y_pred, y_test])

### Building Model using Backward Elimination

import statsmodels.formula.api as sm
X = np.append(values=X, arr=np.ones((50,1)).astype(int), axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# Because this model have the adjust R square value smaller than
# the model above (which mean this model is getting worse).
# So that we take the model above (X[:, [0, 3, 5]]) even this model have a pvalues more than 0.05
# (0.06)
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(1)