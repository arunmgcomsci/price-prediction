# Import the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a synthetic dataset for demonstration
# In a real scenario, you would load your dataset from a file or database.
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Feature (e.g., number of bedrooms)
y = np.array([1, 2, 2.8, 3.5, 4, 5, 5.5, 6, 7, 7.5])  # Target (e.g., house price)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error as a measure of the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Now, you can use the trained model to make predictions on new data.
# For example, to predict the price of a house with 5 bedrooms:
new_data = np.array([5]).reshape(-1, 1)
predicted_price = model.predict(new_data)
print(f"Predicted Price for 5 Bedrooms: {predicted_price[0]}")



