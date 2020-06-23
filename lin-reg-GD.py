# Import packages
# Array Multiplication
import numpy as np
# Graphing
import matplotlib.pyplot as plt

class theta():
    def __init__(self, is_theta_0 = False, val = 0):
        self.is_theta_0 = is_theta_0
        self.val = val

    def compute(self):
        self.temp_val = compute_change(self.val, self.is_theta_0)

    def update(self):
        self.val = self.temp_val

# Generating Data
X_1 = 2 * np.random.rand(100,1)
Y = 6 + 5 * X_1+np.random.randn(100,1)

# Initializing Theta Values, Learning Rate, and Number of Iterations
theta_0, theta_1 = theta(True), theta()
learning_rate = 0.2
iterations = 100

# Theta 0 & 1 Loss History
theta_0_loss_values = []
theta_1_loss_values = []

def compute_change(theta_val, zero_indicator = False):
    '''
    Compute the change to given theta.
    :param theta: Theta Coefficient.
    :param zero_indicator: Used to indicate if variable is theta_0.
    :return: New value of coefficient.
    '''
    if zero_indicator:
        prediction = (theta_1.val * X_1) + theta_0.val
        error = (1 / len(X_1)) * (prediction - Y).sum()
        theta_0_loss_values.append(abs(error))
        change = learning_rate * error
        new_theta = theta_val - change
    else:
        prediction = (theta_1.val * X_1) + theta_0.val
        error = (1 / len(X_1)) * ((prediction - Y)*X_1).sum()
        theta_1_loss_values.append(abs(error))
        change = learning_rate * error
        new_theta = theta_val - change
    return new_theta

for i in range(iterations):
    theta_0.compute()
    theta_1.compute()
    theta_0.update()
    theta_1.update()

# Create Prediction Line for entire data
final_prediction = theta_0.val + (theta_1.val*X_1)
plt.plot(X_1, final_prediction, '-r', label = 'Predictions')
plt.scatter(X_1, Y)
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
