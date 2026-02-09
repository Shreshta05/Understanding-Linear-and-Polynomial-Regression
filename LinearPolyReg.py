#Use Case : Advertising spend vs sales

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#creating synthetic data and adding noise
np.random.seed(42)
X = np.linspace(10_000, 100_000, 200).reshape(-1, 1)
true_sales = 20_000 + 0.8 * X + 0.000005 * (X**2)
noise = np.random.normal(0, 15_000, size=true_sales.shape)
y = true_sales + noise

#visualize the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.title('Synthetic data: Advertising Spend Vs Sales')
plt.savefig('Synthetic Data Advertising spend Vs Sales.png', dpi = 150, bbox_inches = 'tight')
plt.show()

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create linear reg model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train_linear = linear_model.predict(X_train)
y_pred_test_linear = linear_model.predict(X_test)

train_mse_linear = mean_squared_error(y_train, y_pred_train_linear)
test_mse_linear = mean_squared_error(y_test, y_pred_test_linear)

# polynomial reg model
degrees = [1, 2, 3, 5, 8]
train_errors = []
test_errors = []

#poly training loop(for more degrees)
for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    y_pred_train_poly = poly_model.predict(X_train_poly)
    y_pred_test_poly = poly_model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_pred_train_poly))
    test_errors.append(mean_squared_error(y_test, y_pred_test_poly))

#plot the model fits(underfitting / overfitting)
X_plot = np.linspace(10_000, 100_000, 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.3, label = 'Actual Data')

for d in degrees:
    poly = PolynomialFeatures(degree = d, include_bias=False)
    X_plot_poly = poly.fit_transform(X_plot)

    poly_model = LinearRegression()
    poly_model.fit(poly.fit_transform(X_train), y_train)

    plt.plot(X_plot, poly_model.predict(X_plot_poly), label=f"Degree {d}")

plt.xlabel('Advertising spend')
plt.ylabel('Sales')
plt.title('Polynomial Regression Fits')
plt.legend()
plt.savefig('PolyRegFits.png', dpi = 150, bbox_inches = 'tight')
plt.show()

#plot the training and testing errors
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker = '*', color = 'red', label = 'Training Errors')
plt.plot(degrees, test_errors, marker = '*', color = 'green', label = 'Testing Errors')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training Vs Testing Errors')
plt.legend()
plt.savefig('TrainTestErrors.png', dpi = 150, bbox_inches = 'tight')
plt.show()