# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries and initialize the dataset with population (X) and profit (y).
2.  Initialize parameters θ₀ and θ₁, set learning rate (α) and number of iterations.
3.  Compute predicted values using the hypothesis function h(x)=θ0+θ1xh(x) = θ₀ + θ₁xh(x)=θ0​+θ1​x.
4.  Calculate error and gradients for θ₀ and θ₁.
5.  Update parameters using gradient descent and repeat until convergence.
6.  Use final parameters to predict profit and display the result.


## Program:
# Program to implement the linear regression using gradient descent.
# Developed by:  VIJIYALAKSHMI A
# RegisterNumber: 212225240185 
```
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Dataset (Population vs Profit)
# Example dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])   # Population (in 10,000s)
y = np.array([1.5, 2.0, 2.5, 3.5, 3.0, 4.5, 5.0, 6.0, 6.5])  # Profit

m = len(X)

# Step 2: Initialize parameters
theta0 = 0   # Intercept
theta1 = 0   # Slope
alpha = 0.01 # Learning rate
iterations = 1000

# Step 3: Gradient Descent
for i in range(iterations):
    y_pred = theta0 + theta1 * X
    
    # Compute gradients
    d_theta0 = (1/m) * np.sum(y_pred - y)
    d_theta1 = (1/m) * np.sum((y_pred - y) * X)
    
    # Update parameters
    theta0 = theta0 - alpha * d_theta0
    theta1 = theta1 - alpha * d_theta1

# Step 4: Final model parameters
print("Intercept (theta0):", theta0)
print("Slope (theta1):", theta1)

# Step 5: Prediction
pop = 7.5
profit_pred = theta0 + theta1 * pop
print(f"Predicted profit for population {pop} = {profit_pred:.2f}")

# Step 6: Plot graph
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, theta0 + theta1 * X, color='red', label="Regression Line")
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid()
plt.show()
```
## Output:
<img width="839" height="650" alt="Screenshot 2026-04-27 141920" src="https://github.com/user-attachments/assets/4c1f4bf1-8351-4ed0-a2d4-300dbf40300a" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
