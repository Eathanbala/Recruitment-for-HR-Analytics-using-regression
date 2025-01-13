#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Recruitment Pipeline Optimization for HR Analytics
# Using Manual Implementation Without Built-in Functions (Regression Model)

# Sample Dataset
data = [
    {"EmpID": "RM297", "Age": 18, "AgeGroup": "18-25", "Attrition": "Yes", "BusinessTravel": "Travel_Rarely"},
    {"EmpID": "RM302", "Age": 18, "AgeGroup": "18-25", "Attrition": "No", "BusinessTravel": "Travel_Rarely"},
    {"EmpID": "RM458", "Age": 18, "AgeGroup": "18-25", "Attrition": "Yes", "BusinessTravel": "Travel_Frequently"},
    {"EmpID": "RM728", "Age": 18, "AgeGroup": "18-25", "Attrition": "No", "BusinessTravel": "Non-Travel"}
]

# Step 1: Encode Categorical Data Manually
def encode_categorical(data, column):
    unique_values = []
    for row in data:
        if row[column] not in unique_values:
            unique_values.append(row[column])
    
    for row in data:
        row[column] = unique_values.index(row[column])
    return unique_values

# Encode columns
attrition_labels = encode_categorical(data, "Attrition")
business_travel_labels = encode_categorical(data, "BusinessTravel")

# Step 2: Prepare Features and Labels
X = []  # Features
y = []  # Labels
for row in data:
    X.append([row["Age"], row["BusinessTravel"]])  # Features: Age and BusinessTravel
    y.append(row["Attrition"])                      # Label: Attrition

# Convert y labels to numeric for regression (Yes -> 1, No -> 0)
y = [1 if label == 0 else 0 for label in y]

# Step 3: Implement a Simple Linear Regression Algorithm
def train_linear_regression(X, y, epochs, learning_rate):
    weights = [0] * len(X[0])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute prediction
            prediction = sum(X[i][j] * weights[j] for j in range(len(weights))) + bias
            
            # Calculate error
            error = y[i] - prediction

            # Update weights and bias using gradient descent
            for j in range(len(weights)):
                weights[j] += learning_rate * error * X[i][j]
            bias += learning_rate * error

    return weights, bias

# Train the Linear Regression model
weights, bias = train_linear_regression(X, y, epochs=1000, learning_rate=0.01)

# Step 4: Prediction
def predict(X, weights, bias):
    predictions = []
    for i in range(len(X)):
        prediction = sum(X[i][j] * weights[j] for j in range(len(weights))) + bias
        predictions.append(prediction)
    return predictions

predictions = predict(X, weights, bias)

# Step 5: Evaluate the Model
# Calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    error_sum = 0
    for i in range(len(y_true)):
        error_sum += (y_true[i] - y_pred[i]) ** 2
    return error_sum / len(y_true)

mse = mean_squared_error(y, predictions)

# Output Results
print("Weights:", weights)
print("Bias:", bias)
print("Predictions:", predictions)
print("Mean Squared Error (MSE):", mse)


# In[ ]:




