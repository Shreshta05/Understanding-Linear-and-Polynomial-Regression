Understanding-Linear-and-Polynomial-Regression

Understanding Linear and Polynomial Regression Using Synthetic Data (Assignment)

This repository explores how Advertising spend influences product Sales using linear and polynomial regression models. Since real-world marketing data is often noisy and hard to obtain, a synthetic dataset is created to simulate realistic advertising budgets and their impact on sales. By gradually increasing model complexity, the project visually demonstrates underfitting, overfitting, and the bias–variance tradeoff, helping build an intuitive understanding of how regression models learn patterns from data and why choosing the right model complexity matters.

Q: What is the loss function used in linear regression?

A: The loss function used in linear regression is the Mean Squared Error (MSE), which measures the average of the squared differences between the predicted values and the actual values. In the advertising spend versus sales problem, MSE quantifies how far the predicted sales are from the true sales for different levels of advertising spend. By squaring the errors, larger prediction mistakes are penalized more heavily, ensuring that the model focuses on reducing significant deviations. The objective of linear regression is to minimize this loss by adjusting the model parameters so that the prediction line or curve fits the data as closely as possible despite the presence of noise.

Q: Why is squared error used in linear regression?

A: Squared error is used in linear regression because it provides a clear and effective way to measure how far the predicted sales values are from the actual sales values for a given advertising spend. By squaring the difference between predicted and actual values, larger prediction errors are penalized more heavily than smaller ones, which encourages the model to avoid large mistakes. Squaring also ensures that all errors are positive, preventing situations where positive and negative errors cancel each other out. In the context of advertising spend versus sales, this helps the model focus on accurately capturing the overall trend rather than allowing a few large deviations to be ignored.

Q: What does minimizing the loss actually mean?

A: Minimizing the loss means finding the model parameters that make the predicted sales values as close as possible to the actual sales values across all advertising spend levels in the dataset. In practice, this involves adjusting the intercept and coefficients so that the average squared difference between predicted and actual sales is reduced. When the loss is minimized, the regression model produces predictions that best represent the underlying relationship between advertising spend and sales, despite the presence of noise and variability in the data.

Q: How do the model parameters influence the loss?

A: The model parameters, such as the intercept and coefficients, directly influence the loss because they determine the shape and position of the regression line or curve used to predict sales from advertising spend. Changing these parameters alters how strongly advertising spend affects the predicted sales values. If the parameters are poorly chosen, the model’s predictions will be far from the actual data points, resulting in a high loss. When the parameters are well chosen, the prediction curve aligns closely with the observed data, reducing prediction errors and therefore lowering the loss value.

Q: Why does training error always decrease with higher polynomial degree?

A: As the polynomial degree increases, the model becomes more flexible and gains the ability to fit more complex patterns in the training data. Each higher degree adds more parameters, allowing the model to bend and adjust itself to reduce the prediction errors on the training set. Because the model is explicitly optimized to minimize training loss, it can always fit the training data at least as well as, and usually better than, a lower-degree model. This is why, in the training error vs degree plot, the training error consistently decreases as the polynomial degree increases.

Q: Why does test error behave differently from training error?

A: Test error behaves differently because it measures how well the model generalizes to unseen data rather than how well it fits the training data. Initially, as polynomial degree increases, the model captures the true non-linear relationship between advertising spend and sales more accurately, causing test error to decrease. However, beyond a certain degree, the model starts fitting noise and random fluctuations present in the training data instead of the underlying trend. This leads to poorer generalization, and the test error begins to increase, even though the training error continues to decrease.

Q: At what point does the model start overfitting, and how can you tell?

A: The model starts overfitting at the polynomial degree where the test error begins to increase while the training error continues to decrease. In the plots, this is visible when the polynomial curve becomes overly wiggly and closely follows individual data points rather than the overall trend. At this point, the training error is very low, but the test error rises, indicating that the model is memorizing noise instead of learning the true relationship between advertising spend and sales.

Q: If you had to choose one polynomial degree, which would it be and why?

A: The optimal polynomial degree is the one that achieves the lowest test error while still producing a smooth and interpretable fit to the data, as this reflects the best balance between bias and variance. From the training and testing error plots, this occurs at a polynomial degree of 2 or 3, where the model is flexible enough to capture the underlying non-linear relationship between advertising spend and sales without fitting noise. At these degrees, the test error is minimized or very close to its lowest value, and the prediction curve follows the overall trend of the data without excessive oscillations. Higher-degree models, although they continue to reduce training error, begin to overfit the data, which is evident from increasing test error and overly complex curves. Therefore, selecting a polynomial degree of 2 or 3 provides the most reliable model with good generalization to unseen data.
