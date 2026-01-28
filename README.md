# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Read the input values X (independent variable) and y (dependent variable).
2. Assume a linear relationship between X and y using the for
        <img width="492" height="366" alt="p" src="https://github.com/user-attachments/assets/11cb3f0b-768d-46f0-a80c-8ae499fecd6f" />

3.  Predict the output values using:
          <img width="172" height="66" alt="pp" src="https://github.com/user-attachments/assets/6af24619-b2f5-4f76-963b-961036e2b1f2" />

          
4.Display the slope 𝑚 and intercept c, and plot:

*Actual data points

*Regression line
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ASWINI D
RegisterNumber:  25018420
*/
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (X = input, y = output)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
```

## Output:
![exp3mlop](https://github.com/user-attachments/assets/d1ba10bf-ac63-44b4-ad92-5148e781abd3)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
