import numpy as np


def sigmoid(x): #used by feedforward function
    return 1.0 / (1 + np.exp(-x))
def sigmoid_derivative(x): # used for back prop
    return x * (1.0 - x)


class NeuralNetwork: #outputs are only affected by weights and biases
    def __init__(self,x,y):
        self.input = x
        ## next we initalize weights as random NumPy Arrays
        self.weights1 = np.random.rand(self.input.shape[1],4) #self.input.shape[1] is a variable array (creates one depending on size of input)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self): #feedforward that calculates predicted output
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # we use sigmoid (activation function) to squash the values between
        # 0 and 1 because that is the desired range for predictions
        self.output = sigmoid(np.dot(self.layer1,self.weights2))

        # next we need a loss function to measure accuracy
        # our goal is to find the best set of weights and biases that minimize loss function
        # BACKPROPAGATION
        # After measuring the error of our prediction aka loss
        # we must find a way to propagate the error back and updadte weights and biases
        # In order to know the approiate amount to adjust weights and biases, we need
        # to know the derivate of the loss function with respect to our biases and weights
        # if we have the derivative we can update the weights and biases by increasing
        # or reducing with it. This is known as gradient descent

    def backprop(self):  #loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*
                                            sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.layer1.T, (2 * (self.y - self.output) *
                                            sigmoid_derivative(self.output),
                                            self.weights2.T)*sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "_main_":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[0]])
    nn = NeuralNetwork(X,y)
    for i in range(1500):
        nn.feedforward()
        nn.backprop()
    print(nn.output)
        

### WORKING WITH DATASETS
import pandas as pd

# df = pd.read_csv("name of file.csv") #imports as a DataFrame
# print(df.info()) #gives insight into num of cols & rows along with names of cols
# print(df.describe()) #gives mean, std, mind, max, 25%, etc
# print(df.head(10)) #outputs the first 10 rows of data
## you can also specify specifc constraints to only print certain rows/cols
#Ex: df2 - df.loc[df['sepal_length'] > 5.0, ] selects rows with speal greater than 5
# loc allows access to a group of rows and cols










