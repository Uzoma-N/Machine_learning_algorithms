'''
Machine Learning algorithm for linear regression showing the profit prediction derived from the spend on R&D,
Admin and Marketing
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# data loading
data = pd.read_csv("datasets/startups.csv")
"""
# data analysis
print(data.head())  # to view data
print(data.describe())  # to view summary stats of the data
sns.heatmap(data.corr(), annot=True)  # to evaluate correlation
plt.show()
"""
# to prepare the data for training
x = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]

x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# to train the Linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)
predict_data = pd.DataFrame(data={"Predicted Profit": y_pred.flatten()})
print(predict_data.head())
