#SALES PREDICTION USING PYTHON

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('sales_pred.csv')  # Load the data from a CSV file into a pandas DataFrame

# Data Preprocessing and Exploration
print(data.head())  # Display the first 5 rows of the dataset
print(data.isnull().sum())  # Check for missing values in each column
print(data.describe())  # Get a summary of the numerical columns (mean, min, max, etc.)

# Visualize relationships between features (TV, Radio, Newspaper) and Sales using a pairplot
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()  # Display the pairplot

# Visualize the correlation matrix using a heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")  # Display correlation matrix with annotations
plt.show()  # Show the heatmap

# Split the dataset into features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]  # Independent variables (features)
y = data['Sales']  # Dependent variable (target: Sales)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()  # Create a linear regression model
model.fit(X_train, y_train)  # Fit the model to the training data

# Output the intercept and coefficients of the trained model
print(f"Intercept: {model.intercept_}")  # Display the intercept (constant term)
print(f"Coefficients: {model.coef_}")  # Display the coefficients (slopes for each feature)

# Make predictions on the test set
y_pred = model.predict(X_test)  # Predict the sales on the test set

# Compare predicted sales vs actual sales
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  # Create a DataFrame for comparison
print(comparison_df.head())  # Display the first few rows of actual vs predicted sales

# Evaluate the model using R-squared and RMSE (Root Mean Squared Error)
r2 = r2_score(y_test, y_pred)  # Calculate R-squared score (goodness of fit)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate RMSE (lower is better)

print(f"R-squared: {r2:.2f}")  # Display the R-squared value
print(f"RMSE: {rmse:.2f}")  # Display the RMSE value

# Visualize the comparison of actual vs predicted sales using a scatter plot
plt.scatter(y_test, y_pred, alpha=0.8)  # Scatter plot for actual vs predicted
plt.xlabel("Actual Sales")  # Label for the x-axis
plt.ylabel("Predicted Sales")  # Label for the y-axis
plt.title("Actual vs. Predicted Sales")  # Title of the plot

# Plot a red line for the perfect prediction (where actual = predicted)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")  # Line of perfect prediction
plt.show()  # Display the scatter plot

# Print model coefficients and intercept
print('Coefficients:', model.coef_)  # Display the coefficients (influence of features)
print('Intercept:', model.intercept_)  # Display the intercept (constant term)
