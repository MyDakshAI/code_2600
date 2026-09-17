# Module 02: Training Splits

# ---- Load necessary packages ----
import pandas as pd
import numpy as np
import statsmodels.api as sm
from ISLP.models import summarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots


# ---- Helper functions ----

def predict(X, model):
    '''
    The built-in "get_prediction" tool in ISLP returns an array, so this is
    just a wrapper which converts the result to a dataframe.
    '''
    predictions_df = pd.DataFrame(model.get_prediction(X).predicted, columns=['y_hat'], index=X.index)
    return predictions_df['y_hat']


def mse(y, y_hat):
    '''
    Returns the mean squared error, which we will use to evaluate how well a
    linear regression model fits a dataset.

    The details of this function will be explored further in module 04.
    '''
    # calculate the residual error for each individual record
    resid = y - y_hat
    # square the residual (hence "squared error")
    sq_resid = resid**2
    # calculate the sum of squared errors
    SSR = sum(sq_resid)
    # divide by the number of records to get the mean squared error
    MSE = SSR / y.shape[0]
    return MSE


# ---- Randomly generate a dataset using the equation y = x - 2x^2 ----

# Always specify a seed so that the data can be regenerated
# DON'T change anything in this cell.
seed = 314

# Create a random number generator called rng
rng = np.random.default_rng(seed)

# Use the random number generator to create x- and y-coordinates
x = rng.normal(size=150)
y = x - 2 * x**2 + rng.normal(size=150)


# ---- Create a dataframe we can use to build three different models ----

# We'll need a collection of independent variables. We'll use x, x^2, x^3,
# and a constant (which will be used to calculate the intercept)
new_x = pd.DataFrame(np.column_stack((x**0, x, x**2, x**3)), columns=['intercept', 'x', 'x_sq', 'x_cu'])

# print the dataframe
print(new_x)


# ---- Split the data into train and test ----

# Note that we need to choose a random seed and the percent of records withheld for testing
x_train, x_test, y_train, y_test = train_test_split(new_x,
                                                      y,
                                                      random_state=314159,
                                                      test_size=0.33,
                                                      shuffle=True)


# ---- Graph the training data ----

# It's important to do this after the train/test split is created. We shouldn't look at data that's withheld for testing.
fig, ax = subplots()
ax.scatter(x_train['x'], y_train)
plt.title('Training data')
plt.show()


# ---- Quadratic model (should be appropriate since data is quadratic) ----

model_quad = sm.OLS(y_train, x_train[['intercept', 'x', 'x_sq']])
results_quad = model_quad.fit()
print(summarize(results_quad))

# Get predictions
x_train['pred_sq'] = predict(x_train[['intercept', 'x', 'x_sq']], results_quad)

# Plot predictions as solid green line, along with the original training data
fig, ax = subplots()
ax.scatter(x_train['x'], y_train)
quadratic = x_train[['x', 'pred_sq']].sort_values('x')
ax.plot(quadratic['x'], quadratic['pred_sq'], color='green')
plt.title('Quadratic model fit')
plt.show()


# ---- Underfit model: linear ----

# Fit a linear model
model_lin = sm.OLS(y_train, x_train[['intercept', 'x']])
results_lin = model_lin.fit()
print(summarize(results_lin))

# Get model predictions
x_train['pred_lin'] = predict(x_train[['intercept', 'x']], results_lin)

# Plot predictions as solid *orange* line
fig, ax = subplots()
ax.scatter(x_train['x'], y_train)
linear = x_train[['x', 'pred_lin']].sort_values('x')
ax.plot(linear['x'], linear['pred_lin'], color='orange')
plt.title('Linear (underfit) model fit')
plt.show()


# ---- Overfit model: cubic ----

# Fit a cubic model
model_cubic = sm.OLS(y_train, x_train[['intercept', 'x', 'x_sq', 'x_cu']])
results_cubic = model_cubic.fit()
print(summarize(results_cubic))

# Get predictions
x_train['pred_cu'] = predict(x_train[['intercept', 'x', 'x_sq', 'x_cu']], results_cubic)

# Plot predictions as solid *red* line
fig, ax = subplots()
ax.scatter(x_train['x'], y_train)
cubic = x_train[['x', 'pred_cu']].sort_values('x')
ax.plot(cubic['x'], cubic['pred_cu'], color='red')
plt.title('Cubic (overfit) model fit')
plt.show()


# ---- Discussion: comparing the three models on training data ----
#
# The quadratic model fits the data well and closely tracks the curved
# pattern in the scatter plot, which makes sense since the data was
# generated from a quadratic equation. The linear model is clearly the
# worst on training -- it can't capture the curvature at all, appearing
# as a straight line that misses the bend in the data (underfitting).
# The cubic model looks very similar to the quadratic model, and may
# even hug the training points slightly more closely due to its extra
# flexibility, but the improvement over quadratic (if any) is marginal
# since the true relationship has no cubic term.


# ---- Calculate the errors (MSE) for each model on the training set ----

predictions_lin_train = predict(x_train[['intercept', 'x']], results_lin)
predictions_quad_train = predict(x_train[['intercept', 'x', 'x_sq']], results_quad)
predictions_cubic_train = predict(x_train[['intercept', 'x', 'x_sq', 'x_cu']], results_cubic)

print('mse train linear   :', mse(y_train, predictions_lin_train))
print('mse train quadratic:', mse(y_train, predictions_quad_train))
print('mse train cubic    :', mse(y_train, predictions_cubic_train))


# ---- Calculate the errors (MSE) for each model on the test set ----

predictions_lin_test = predict(x_test[['intercept', 'x']], results_lin)
predictions_quad_test = predict(x_test[['intercept', 'x', 'x_sq']], results_quad)
predictions_cubic_test = predict(x_test[['intercept', 'x', 'x_sq', 'x_cu']], results_cubic)

print('mse test linear   :', mse(y_test, predictions_lin_test))
print('mse test quadratic:', mse(y_test, predictions_quad_test))
print('mse test cubic    :', mse(y_test, predictions_cubic_test))


# ---- Discussion: train vs. test performance ----
#
# The linear model has high MSE on both the training and test sets -- it's
# too inflexible to capture the true quadratic relationship, so it
# performs poorly everywhere. This is the classic signature of
# underfitting (high bias).
#
# The quadratic model has low MSE on both training and test sets, since
# it matches the true structure of the data (recall y = x - 2x^2 + noise).
# This is the well-fit, "just right" model.
#
# The cubic model will likely have training MSE that's about the same as
# (or very slightly lower than) the quadratic model, since the extra
# flexibility lets it fit training data marginally better or fit noise.
# However, its test MSE will be equal to or higher than the quadratic
# model's, since the extra cubic term doesn't reflect any true signal --
# it's just fitting noise in the training data. This gap between low
# training error and higher test error is the signature of overfitting
# (high variance).
#
# Overall: linear is most likely underfit; cubic is most likely overfit;
# quadratic is the best-balanced model.
