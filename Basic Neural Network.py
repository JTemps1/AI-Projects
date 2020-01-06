'''
This is a very basic neural network built from scratch which will take its input as a 4 digit array of 
1s and 0s. An even number of 1s should output a 1, and an odd number a 0. 
As this is a first-time build, I'll go through quite slowly.

A basic neural network consists of input nodes, intermediate layers, and output. Each node needs a set of
weights and biases and an activaton function. A common activation function is the sigmoid function, so to 
kick things off I'll make a quick function to implement that. The IDE I'm using doesn't have standard libraries
built-in, so first thing to do is 'pip install numpy' in the terminal.
'''
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#We'll need the derivative of the sigmoid function too:
def d_sigmoid(x):
    return x * (1 - x)

'''
Lovely. Now we'll define our neural network class, which will need several methods.
For now, the network will have an input layer, an intermidiate layer of 4 nodes, and an output layer.
'''

class Neural_Network:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
    
    '''
    In order for the network to learn, we need feedforward and backpropagation methods. The feedforward
    process simply calculates the predicted output, and then backpropagation evaluates how accurate this is
    and updates the weights and biases accordingly.
    '''

    def feedforward(self):
        #This assumes biases = 0, so each layer is just the previous layer nodes multiplied by their weights
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output
    
    '''
    The loss function essentially tells us how 'wrong' the feedforward prediction was. Backpropagation works 
    by findng the minimum of the loss function. The calculus can be quite meaty, but all we need is the
    derivative of the loss function as a function of weights and biases. If we have this, we can find its
    minimum => minimise the loss function.
    '''
    
    def backprop(self):
        weight_change2 = np.dot(self.layer1.T, (2*(self.y - self.output) * d_sigmoid(self.output)))
        weight_change1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * d_sigmoid(self.output), 
                                self.weights2.T) * d_sigmoid(self.layer1)))
        
        #Now update the weights:
        self.weights1 += weight_change1
        self.weights2 += weight_change2
    
    '''
    Lastly, we want a function to actually train the network. This only needs to make the feedforward and
    backpropagation happen.
    '''

    def train(self, x, y):
        self.output = self.feedforward()
        self.backprop()

'''
Now all that has been declared, we can set about training our network. We'll need some training data to begin
with, so first job is to declare some inputs and their respective outputs:
'''
X = np.array(([0,0,0,0],
            [0,1,0,0],
            [1,0,1,0],
            [0,0,0,1],
            [1,1,1,1],
            [0,1,1,1]), dtype=float)
Y = np.array(([1],[0],[1],[0],[1],[0]), dtype=float)

'''
All there is left to do now is create the network and train it! I'll use a loop that trains it 100 times
and prints out the training progress every 10 iterations. This isn't very visually satisfying, so I'll also 
plot the loss function for each iteration, so we can see how the network converges on the optimal weights.
To do this, we'll need to import pyplot from the matplotlib package.
'''
Network1 = Neural_Network(X, Y)

import matplotlib.pyplot as plt 

losses = []
for n in range(100):
    if n % 10 == 0 or n == 0:
        print('\nIteration number ' + str(n) + '\n')
        print('Input: ' + str(X) + '\n')
        print('Expected output: ' + str(Y) + '\n')
        print('Actual output: ' + str(Network1.output) + '\n')
        print ("Loss: " + str(np.mean(np.square(Y - Network1.feedforward()))) + '\n')

    losses.append(np.mean(np.square(Y - Network1.feedforward())))
    Network1.train(X, Y)

plt.plot(range(100), losses)
