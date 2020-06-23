# Import packages
# Array Multiplication
import numpy as np
# Graphing
import matplotlib.pyplot as plt

class theta():
    def __init__(self,variable_link = 0, is_theta_0 = False, val = 0):
        self.is_theta_0 = is_theta_0
        self.val = val
        self.variable_link = variable_link

    def compute(self):
        self.temp_val = compute_change(self.val, self.is_theta_0)

    def update(self):
        self.val = self.temp_val

# Generating Data
X_1 = 2 * np.random.rand(100,1)
X_2 = 2 * np.random.rand(100,1)
Y = np.random.randint(1,8) + 4 * X_1 + np.random.randn(100,1) + (3* X_2 + np.random.randn(100,1))

# Initializing Theta Objects
# Is there a way to automate this for more variables?
theta_0, theta_1 = theta(True), theta()
theta_2 = theta_2
theta_list = [theta_0, theta_1, theta_2]

theta_dict = {theta_0:1, theta_1:len(X_1), theta_2:len(X_2)}

# Gradient Descent parameters
learning_rate = 0.1
epochs = 50
samples = len(X_1)

# Track Loss History
loss = 0
loss_history = []

def compute_change(theta_val, zero_indicator = False):
    '''
    Compute the change to given theta.
    :param theta_val: Theta Coefficient.
    :param zero_indicator: Used to indicate if variable is theta_0.
    :return: New value of coefficient.
    '''
    if zero_indicator:
        prediction = predict()
        error = (1 / samples) * (prediction - Y).sum()
        change = learning_rate * error
        new_theta = theta_val - change
    else:
        prediction = predict()
        error = (1 / samples) * ((prediction - Y)*X_1).sum()
        change = learning_rate * error
        new_theta = theta_val - change

    global loss
    loss += abs(error)
    return new_theta

def predict():
    return (theta_1.val * X_1) + theta_0.val

for epoch in range(epochs):
    loss = 0
    for z in theta_list:
        z.compute()
    for y in theta_list:
        y.update()
    loss_history.append(loss)

# Create Prediction Line for entire data
final_prediction = predict()
plt.plot(X_1, final_prediction, '-r', label = 'Predictions')
plt.scatter(X_1, Y)
plt.title('Predictions vs. Real Values')
plt.show()

# Plot Global Loss
plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Global Loss')
plt.show()