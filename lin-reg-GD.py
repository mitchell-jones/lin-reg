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
        self.temp_val = compute_change(self.val, self.variable_link)

    def update(self):
        self.val = self.temp_val

# Initializing Theta Objects for Univariate Analysis
# Is there a way to automate this for more variables?
theta_0, theta_1 = theta(0), theta(1)

# Find out if we are doing univariate or multivariate regression.
desired_variables = int(input('Do you want univariate(1) or multivariate(2) Linear Regression?')) + 1
variable_coefficients = [theta_0, theta_1]

# Generating Data
samples = 100
X_1 = 2 * np.random.rand(samples,1)
variable_list = [1, X_1]

if desired_variables == 2:
    Y = np.random.randint(1, 8) + (4 * X_1 + np.random.randn(100, 1))
    def predict():
        return (theta_1.val * X_1) + theta_0.val
else:
    theta_2 = theta(2)
    variable_coefficients.append(theta_2)
    X_2 = 2 * np.random.rand(samples, 1)
    variable_list.append(X_2)
    def predict():
        return (theta_1.val * X_1) + (theta_2.val*X_2) + theta_0.val
    Y = np.random.randint(1, 8) + (4 * X_1 + np.random.randn(100, 1)) + (3 * X_2 + np.random.randn(100, 1))

# Gradient Descent parameters
learning_rate = 0.01
epochs = 250

# Function for computing change to theta
def compute_change(theta_val, variable_index):
    '''
    Compute the change to given theta.
    :param theta_val: Theta Coefficient.
    :param zero_indicator: Used to indicate if variable is theta_0.
    :return: New value of coefficient.
    '''
    if variable_index == 0:
        prediction = predict()
        error = (1 / samples) * (prediction - Y).sum()
        change = learning_rate * error
        new_theta = theta_val - change
    else:
        prediction = predict()
        error = (1 / samples) * ((prediction - Y)*variable_list[variable_index]).sum()
        change = learning_rate * error
        new_theta = theta_val - change

    global loss
    loss += abs(error)
    return new_theta

# Track Loss History
loss = 0
loss_history = []

print('Beginning Gradient Descent')
for epoch in range(epochs+1):
    loss = 0
    for z in variable_coefficients:
        z.compute()
    for y in variable_coefficients:
        y.update()
    print("Epoch %s | Loss of %s" % (epoch, round(loss, 3)))
    loss_history.append(loss)
print('Gradient Descent Finished')

# Create Prediction Line for entire data if UniVariate
final_prediction = predict()
if len(variable_coefficients) < 3:
    plt.plot(X_1, final_prediction, '-r', label = 'Predictions')
    plt.scatter(X_1, Y)
    plt.title('Predictions vs. Real Values')
    plt.show()

# Plot Loss History
plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss vs. Epochs')
plt.show()

RMSE = np.sqrt(((final_prediction - Y)**2).sum() / samples)
print('Final RSME:', RMSE)