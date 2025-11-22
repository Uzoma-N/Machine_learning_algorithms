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
from sklearn.metrics import r2_score

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
x_df = data[["R&D Spend", "Administration", "Marketing Spend"]]
y_df = data["Profit"]

# Capture names before conversion for plotting labels
feature_names = x_df.columns.tolist()
y_name = y_df.name

x = x_df.to_numpy()
y = y_df.to_numpy()
y = y.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# to train the Linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

# Display a sample of the predictions
predict_data = pd.DataFrame(data={"Predicted Profit": y_pred.flatten()})
print(predict_data.head())

# Set a professional style for the plots
sns.set_theme(style="whitegrid", palette="deep")

# ----------------------------------------------------
# VISUAL 1: Actual vs. Predicted Plot (Performance Visual)
# ----------------------------------------------------
plt.figure(figsize=(10, 8))

# Scatter plot of Actual Y (test) vs. Predicted Y (pred)
sns.scatterplot(
    x=ytest.flatten(),
    y=y_pred.flatten(),
    s=150,
    alpha=0.7,
    color='#0072B2',
    label='Predicted Points'
)

# Plot the ideal y=x line
max_val = max(ytest.max(), y_pred.max())
min_val = min(ytest.min(), y_pred.min())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linestyle='--',
    linewidth=3,
    label='Ideal Prediction Line (Actual = Predicted)'
)

# Calculate R-squared score
r_sq = r2_score(ytest, y_pred)

# Add professional enhancements
plt.title(f'Model Performance: Actual {y_name} vs. Predicted {y_name}',
          fontsize=18, fontweight='bold', color='#333333', pad=20)
plt.xlabel(f'Actual {y_name} (Test Data)', fontsize=14)
plt.ylabel(f'Predicted {y_name}', fontsize=14)
plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)

# Annotate R-squared score
plt.text(0.05, 0.95, f'R-squared Score: {r_sq:.4f}',
         transform=plt.gca().transAxes,
         fontsize=14,
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='#CCCCCC', boxstyle='round,pad=1'))

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# VISUAL 2: Coefficient Bar Chart (Feature Importance Visual)
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

# Prepare coefficient data
coefficients = model.coef_[0]
intercept = model.intercept_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient_Value': coefficients
}).sort_values(by='Coefficient_Value', ascending=False)

# Define colors based on the sign of the coefficient
bar_colors = ['#D55E00' if c < 0 else '#009E73' for c in coef_df['Coefficient_Value']]

# Create the bar plot
sns.barplot(
    x='Coefficient_Value',
    y='Feature',
    data=coef_df,
    palette=bar_colors
)

# Add text labels for the exact coefficient values
for index, row in coef_df.iterrows():
    plt.text(row['Coefficient_Value'], index, f' {row["Coefficient_Value"]:.3f}',
             color='black' if row['Coefficient_Value'] > 0 else 'white',
             va='center', fontweight='bold')


# Add professional enhancements
plt.title('Feature Impact (Model Coefficients)',
          fontsize=18, fontweight='bold', color='#333333', pad=20)
plt.xlabel('Coefficient Value (Impact on Profit)', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# Display the Intercept value separately
plt.figtext(0.15, 0.90, f'Intercept (Base {y_name}): ${intercept:,.0f}',
            fontsize=12, style='italic', color='#666666',
            bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()