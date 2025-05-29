#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the dataset
# Replacing with actual dataset path or URL
url = "C:/Users/Lenovo/Downloads/Housing.csv"
df = pd.read_csv(url)

# Displaying the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Checking for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

#one-hot encoding categorical vaiables
df = pd.get_dummies(df, drop_first=True)

# Defining features (X) and target (y)
X = df.drop('price', axis=1)   # 'price' is the median value of owner-occupied homes
y = df['price']

# Spliting the data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Coefficients
print("\nModel Coefficients:")
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)

print(f"\nIntercept: {model.intercept_}")



