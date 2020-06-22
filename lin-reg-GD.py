# Import packages
# Array Multiplication
import numpy as np
# Graphing
import matplotlib.pyplot as plt

# Generating Data
predictor_col = 2 * np.random.rand(100,1)
result_col = 6 + 5 * predictor_col+np.random.randn(100,1)

# Initializing Theta Values, Learning Rate, and Number of Iterations
theta_0, theta_1 = 0,0
learning_rate = 0.1
iterations = 50

# Theta 0 & 1 Loss History
theta_0_loss_values = []
theta_1_loss_values = []

def compute_change(theta, zero_indicator = False):
    '''
    Compute the change to given theta.
    :param theta: Theta Coefficient.
    :param zero_indicator: Used to indicate if variable is theta_0.
    :return: New value of coefficient.
    '''
    if zero_indicator:
        prediction = (theta_1 * predictor_col) + theta_0
        error = (learning_rate / len(predictor_col)) * (prediction - result_col).sum()
        theta_0_loss_values.append(abs(error))
        new_theta = theta - error
    else:
        prediction = (theta_1 * predictor_col) + theta_0
        error = (learning_rate / len(predictor_col)) * ((prediction - result_col)*predictor_col).sum()
        theta_1_loss_values.append(abs(error))
        new_theta = theta - error
    return new_theta

for i in range(iterations):
    # For every iteration, update theta_0 and theta_1 simultaneously
    theta_0, theta_1 = compute_change(theta_0, True), compute_change(theta_1)

# Create Prediction Line for entire data
final_prediction = theta_0 + (theta_1*predictor_col)
plt.plot(predictor_col, final_prediction, '-r', label = 'Predictions')
plt.scatter(predictor_col, result_col)
plt.title('Predictions vs. Real Values')
plt.show()

# Plot theta_0 loss
plt.plot(theta_0_loss_values)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Theta_0 Loss')
plt.show()

# Plot theta_0 Loss
plt.plot(theta_1_loss_values)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Theta_1 Loss')
plt.show()