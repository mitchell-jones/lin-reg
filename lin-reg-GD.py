# Import packages
# Array Multiplication
import numpy as np
# Graphing
import matplotlib.pyplot as plt
print('''Welcome to the interactive demo for Gradient Descent parameter tuning.\n
Here, you'll get the chance to change learning rate and epochs to fit a line of best fit for a linear regression problem!
''')


# Set Random Seed
np.random.seed(1)

class theta():
    def __init__(self, variable_index = 0, val = 0):
        self.val = val
        self.variable_index = variable_index

    def compute(self):
        self.temp_val = compute_change(self.val, self.variable_index)

    def update(self):
        self.val = self.temp_val

# Find out if we are doing univariate or multivariate regression.
while True:
    try:
        desired_variables = int(input('How many variables do you want? Acceptable answers are in integer form.'))
        break
    except ValueError:
        print('That is not a valid integer')

# Initializing Theta Objects for Univariate Analysis
variable_coefficients = [theta(i) for i in range(desired_variables + 1)]

# Number of Data Points
samples = 100

# Initialize array with ones for multiplication with theta_0
variable_list = [np.ones(shape=(samples,1))]

# Theta_0 value or b in y=mx+b
Y = np.random.randint(1, 8)
# create data distributions, and store them so we can check our variables against real values as we go
for i in range(len(variable_coefficients)-1):
    new_X = 2 * np.random.rand(samples,1)
    variable_list.append(new_X)
    Y += np.random.randint(1,4) * new_X + np.random.randn(samples, 1)

# this returns an array (weights times our variables) that is our current set of predictions
def predict():
    # initializes empty array to add to
    prediction_array = np.zeros((samples,1))
    for i in variable_coefficients:
        prediction_array = np.add(prediction_array, i.val * variable_list[i.variable_index])
    return prediction_array

loss = 0
# Function for computing change to theta
def compute_change(current_val, variable_index):
    '''
    Compute the change to given theta.
    :param current_val: Theta Coefficient.
    :param zero_indicator: Used to indicate if variable is theta_0.
    :return: New value of coefficient.
    '''
    if variable_index == 0:
        prediction = predict()
        error = (1 / samples) * (prediction - Y).sum()
        change = learning_rate * error
        new_theta = current_val - change
    else:
        prediction = predict()
        error = (1 / samples) * ((prediction - Y) * variable_list[variable_index]).sum()
        change = learning_rate * error
        new_theta = current_val - change

    global loss
    loss += abs(error)
    return new_theta

def fit():
    # Track Loss History
    loss_history = []
    print('Beginning Gradient Descent')
    for epoch in range(epochs+1):
        global loss
        loss = 0
        for y in variable_coefficients:
            y.compute()
        for z in variable_coefficients:
            z.update()
        print("Epoch %s | Loss of %s" % (epoch, round(loss, 3)))
        loss_history.append(loss)
    print('Gradient Descent Finished')

    # Create Prediction Line for entire data if single variable
    final_prediction = predict()
    if desired_variables == 1:
        plt.plot(variable_list[1], final_prediction, '-r', label = 'Predictions')
        plt.scatter(variable_list[1], Y)
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

# Gradient Descent parameters
while True:
    learning_rate = 0.01
    epochs = 250
    print('\nTime to tune your parameters - Here are your previous parameters:\n'
      'Learning Rate: %s\n'
      'Epochs: %s' % (learning_rate, epochs))
    select_differents = input('Would you like to change them? (y/n) q: quit').upper()
    if select_differents == 'Y':
        try:
            learning_rate = float(input('What would you like to set the learning rate to? \n'
                                        'Answer must be in float form.'))
        except:
            print('That is not a valid learning rate, please enter a valid float.')
        try:
            epochs = int(input('What would you like to set the epochs to? \n'
                                        'Answer must be in integer form.'))
        except:
            print('That is not a valid value for epochs, please enter an integer.')
    elif select_differents == 'Q':
        print('Thanks for using the Interactive Demo for Gradient Descent Tuning!')
        quit()
    fit()