import numpy as np # import
import matplotlib.pyplot as plt

# Jack Eckert - 4/04/2025

# ACTIVATION FUNCTIONS -----------------------------------------------------------------------------------------------

'''
ACTIVATION FUNCTIONS

Args:
x -> input
derivative -> calculates the derivative at the given input if True

Returns: float/array
'''

def sigmoid(x, derivative=False):
    s = 1 / (1 + np.exp(-x))
    if derivative:
        return s * (1 - s)
    return s
    
def ReLu(x, derivative=False):
    if derivative:
        return bool(x > 0)
    return max(0, x)
    
def leakyReLu(x, derivative=False):
    if derivative:
        return 1 if (x > 0) else 0.1
    return max(0.1 * x, x)
    
def softmax(x, derivative=False):
    if derivative:
        return 1
    return np.exp(x) / np.sum(np.exp(x))

# COST FUNCTIONS -----------------------------------------------------------------------------------------------------

'''
COST FUNCTIONS

Args:
x -> input
derivative -> calculates the derivative at the given input if True

Returns: float
'''

def MSE(obs, true, derivative=False):
    if derivative:
        return 2 * (obs - true)
    return (obs - true) ** 2
    
def catCrossEntropy(obs, true, derivative=False):
    if derivative:
        return obs - true
    print(obs)
    obs[obs == 0] = 1e-10
    print(obs, true)
    print(-(true * np.log(obs)))
    exit()
    return -(true * np.log(obs))

_fullArrays = {softmax, catCrossEntropy}

# MISC ---------------------------------------------------------------------------------------------------------------
def createDataset(inputArr, outputArr):

    '''
    Takes a 1d or 2d numpy array of inputs (as the rows) and a 1d or 2d numpy array of outputs (as the rows) and returns
    a list of Datapoint objects

    Args:
    inputArr -> array of inputs, 1d or 2d numpy array of numerics
    outputArr -> array of outputs, 1d or 2d numpy array of numerics

    Returns: list of Datapoint objects
    '''

    lst = []

    for input, output in zip(inputArr, outputArr):
        lst.append(Datapoint(input, output))

    return lst


def formatData(array, separator: int, outputFirst = False):

    '''
    Splits each row of a 2D numpy array into inputs and outputs, then returns an array of datapoint objects using them.
    Returned array should be equal in size to number of rows in the input array.

    Use only if you have one numpy array consisting of both inputs and outputs. If you have two seperate arrays, one for inputs
    one for outputs, use neural.createDataset

    Args:
    array -> 2D array of your data
    separator -> index of start of the output.
    outputFirst -> flips input and output. Use if your output comes before input in your data.

    Returns: an array of Datapoint objects
    '''

    lst = []
    for row in array:
        if outputFirst:
            lst.append(Datapoint(row[separator:], row[:separator]))
        else:
            lst.append(Datapoint(row[:separator], row[separator:]))
    return np.array(lst)

def save(neuralNetwork, filepath):

    '''
    Saves a NeuralNetwork object as a .npz file (numpy archive). First array is shape, second is cost function, and then every 
    three after that are the weights array, bias array, and activation function of a specific layer.

    Args:
    neuralNetwork -> NeuralNetwork object to be saved.
    filepath -> filepath to save NeuralNetwork object as.

    Returns: None
    '''

    lst = [np.array(neuralNetwork.shape), neuralNetwork.trueCost]

    for layer in neuralNetwork.layers:
        lst.extend([layer.weightsArray, layer.biasArray, layer.trueAct])

    np.savez(filepath, *lst)

def load(filepath):

    '''
    Loads and returns a NeuralNetwork object from the specified filepath (.npz format)

    Args:
    filepath -> filepath from where to load the file

    Returns: NeuralNetwork object
    '''

    a = np.load(filepath, allow_pickle=True)

    nn = NeuralNetwork(tuple(a["arr_0"]), costFunction=a["arr_1"].item())

    for i, layer in enumerate(nn.layers):
        j = i * 3
        layer.setWeights(a[f"arr_{j + 2}"])
        layer.setBiases(a[f"arr_{j + 3}"])
        layer.setActivationFunction(a[f"arr_{j + 4}"].item())

    return nn


# CLASSES  -----------------------------------------------------------------------------------------------------------
class Datapoint():
    def __init__(self, input, output):

        '''
        Initializes datapoint object

        Args:
        input -> input of the datapoint
        output -> expected output of the datapoint
        '''

        self.inputs = np.array(input)
        self.outputs = np.array(output)

class _Layer():

    '''
    Represents a layer of the neural network. A 'layer' in this implementation is a set of nodes and their incoming weights.

    Attributes:
    weightsArray -> matrix of incoming weights as a numpy array, 2d numpy array of floats
    biasArray -> vector of layer's biases as a numpy array, 1d numpy array of floats
    numNodesOut -> # of nodes in layer, int
    numNodesIn -> # of nodes in preceding layer, int
    activationFunction -> vectorized activation function for the layer, vectorized pyfunc
    trueAct -> activation function as a function, pyfunc
    outputs -> values of the outputs of the nodes, 1d numpy array of floats
    preActs -> values of the outputs before being inputted into the activation function, 1d numpy array of floats
    inputs -> values of the inputs to the layer. Equal to the output of the previous layer, 1d numpy array of floats
    weightsGradient -> current calculated matrix of gradients to add to the weights, 2d numpy array of floats
    biasGradient -> current calculated vector of gradients to add to the biases, 1d numpy array of floats

    Public Methods:
    setWeights -> sets the weights of a layer to a given input matrix. Must be a 2d numpy array
    setBiases -> sets the biases of a layer to a given input vector. Must be a 1d numpy array
    updateWeights -> adds the current gradient matrix to the weights matrix and then resets it to zero
    updateBiases -> adds the current gradient vector to the bias vector and then resets it to zero
    setActivationFunction -> sets the activation function of the layer to the input.
    '''

    def __init__(self, numNodesIn: int, numNodesOut: int, activationFunction, initializeMode):

        '''
        Initializes layer object.

        Args:
        numNodesIn -> # of input nodes. Should be the same as the # of nodes of the preceding layer.
        numNodesOut -> # of nodes in layer.
        activationFunction -> activation function for the layer
        initializeMode -> specifications for how to initialize the weights
            'r' -> initialize using the recommended initialization based on the activation function
            'n' -> initialize using a standard normal Xavier initialization
            'u' -> initialize using a standard uniform Xavier initialization
            'h' -> initialize using a standard He initialization
            'l' -> initialize using a standard LeCun initialization
        '''
        
        self.biasArray = np.zeros(shape=numNodesOut)
        self.numNodesOut = numNodesOut
        self.numNodesIn = numNodesIn
        if activationFunction in _fullArrays:
            self.activationFunction = activationFunction
        else:
            self.activationFunction = np.vectorize(activationFunction, excluded=["derivative"])
        self.trueAct = activationFunction
        self.outputs = None
        self.preActs = None
        self.inputs = None
        self.weightsGradient = np.zeros(shape=(numNodesIn, numNodesOut))
        self.biasGradient = np.zeros(shape=(numNodesOut))

        self._initializeViaMode(initializeMode, numNodesIn, numNodesOut)

    def _initializeViaMode(self, mode, numNodesIn, numNodesOut):
        if mode == "r":
            if self.trueAct == sigmoid:
                mode = "n"
            elif self.trueAct == ReLu:
                mode = "h"
            elif self.trueAct == leakyReLu:
                mode = "l"
            else:
                mode = "u"

        if mode == "n":
            self.weightsArray = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2 / (numNodesIn + numNodesOut))
        elif mode == "u":
            self.weightsArray = (np.random.random(size=(numNodesIn, numNodesOut)) - 0.5) * np.sqrt(6 / (numNodesIn + numNodesOut)) * 2
        elif mode == "h":
            self.weightsArray = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(2 / (numNodesIn))
        elif mode == "l":
            self.weightsArray = np.random.randn(numNodesIn, numNodesOut) * np.sqrt(1 / (numNodesIn))          

    def _calculateOutputs(self, inputs):
        self.inputs = np.array(inputs)
        self.preActs = (np.matmul(self.inputs, self.weightsArray) + self.biasArray)
        self.outputs = self.activationFunction(self.preActs, derivative=False)
        return self.outputs
    
    def setWeights(self, newWeights):
        self.weightsArray = newWeights

    def setBiases(self, newBiases):
        self.biasArray = newBiases

    def _layerCost(self, actualOutput, expectedOutput, costFunction):
        return costFunction(actualOutput, expectedOutput, derivative=False)
    
    def updateWeights(self, learnRate, batchSize):
        self.weightsArray += learnRate * self.weightsGradient / batchSize
        self.weightsGradient = np.zeros(shape=(self.numNodesIn, self.numNodesOut))

    def updateBiases(self, learnRate, batchSize):
        self.biasArray += learnRate * self.biasGradient / batchSize
        self.biasGradient = self.biasGradient = np.zeros(shape=(self.numNodesOut))

    def setActivationFunction(self, newAct):
        self.trueAct = newAct
        if newAct in _fullArrays:
            self.activationFunction = newAct
            return
        
        self.activationFunction = np.vectorize(newAct, excluded=["derivative"])
    
class NeuralNetwork():

    '''
    represents a neural network as a single object. Contains a list of layers within it. The input layer is not considered a layer, 
    instead acting merely as the input of the actual first layer, either the hidden layer or the output layer depending on size.
    e.g. a neural network consisting of 1 hidden layer will contain two layer objects: the hidden layer with its input weights & biases,
    and the output layer with the same.

    Attributes:
    self.layers -> list of layer objects which make up the neural network, list
    self.size -> # of layers in the neural network. In this case, the input layer DOES count as a layer, int
    self.shape -> a tuple consisting of the size of the layers. e.g. a network with 2 inputs, 1 hidden layer with 2 nodes, and 1 output
    would be (2, 2, 1)
    costFunction -> vectorized cost function for the network, vectorized pyfunc
    trueCost -> cost function as a function, pyfunc

    Public Methods:
    calculateOutput -> takes a datapoint object and returns its respective output
    setOutputActivationFunction -> sets the output layer's activation function to the inputted value
    evaluatePoint -> takes a datapoint object and returns True if the program predicted the correct output, otherwise returns False
    evaluate -> takes an iterable of datapoint objects and returns the total count of correctly predicted outputs
    cost -> takes a datapoint object and returns its cost
    train -> trains the model
    '''

    def __init__(self, shape: tuple, activationFunction = ReLu, costFunction = catCrossEntropy, initializeMode = "r"):

        '''
        Initializes neural network object.

        IMPORTANT: If you put ReLu or a related function as your inputted activation function, the initialization will
        automatically set your output layer's activation function to sigmoid. If you wish to change this, use the 
        setOutputActivationFunction method with whatever you want the output activation function to be.

        Args:
        shape -> tuple representing the # of nodes in each layer, tuple
        activationFunction -> starting activation function for the network, pyfunc
        costFunction -> costFunction for the network, pyfunc
        initializeMode -> specifications for how to initialize the weights and biases, char (string)
        '''

        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(_Layer(shape[i], shape[i + 1], activationFunction, initializeMode))
        self.size = len(self.layers) + 1
        self.shape = shape
        if costFunction in _fullArrays:
            self.costFunction = costFunction
        else:
            self.costFunction = np.vectorize(costFunction, excluded=['derivative'])
        self.trueCost = costFunction

        if activationFunction in [ReLu, leakyReLu]:
            self.setOutputActivationFunction(softmax)
 
    def calculateOutput(self, datapoint):

        # allows for inputs of both datapoints and standard iterables
        if isinstance(datapoint, Datapoint):
            put = datapoint.inputs
        else:
            put = datapoint

        for layer in self.layers:
            put = layer._calculateOutputs(put)
        return put
    
    def setOutputActivationFunction(self, activationFunction, reinitialize=True):
        self.layers[-1].trueAct = activationFunction
        if activationFunction in _fullArrays:
            self.layers[-1].activationFunction = activationFunction
        else:
            self.layers[-1].activationFunction = np.vectorize(activationFunction, excluded=['derivative'])
        
        if reinitialize:
            self.layers[-1]._initializeViaMode('r', self.layers[-1].numNodesIn, self.layers[-1].numNodesOut)
    
    def evaluatePoint(self, datapoint: Datapoint):
        output = self.calculateOutput(datapoint)
        if len(output) == 1:
            return int(round(output.item())) == datapoint.outputs.item()
        return np.argmax(output, 0) == np.argmax(datapoint.outputs, 0)
    
    def evaluate(self, dataset):
        total = 0
        for datapoint in dataset:
            total += int(self.evaluatePoint(datapoint))
        return total
    
    def cost(self, datapoint: Datapoint, returnTotal=True):
        output = self.calculateOutput(datapoint)
        oLayer = self.layers[-1]

        if returnTotal:
            return np.sum(oLayer._layerCost(output, datapoint.outputs, self.costFunction))
        else:
            return oLayer._layerCost(output, datapoint.outputs, self.costFunction)
        
    def _getEndVals(self, datapoint: Datapoint):

        '''
        Returns the "end values" of the network. "end values" are defined as the derivatives of the cost function with respect to the 
        post-activation values of the end layer times the derivatives of the activation function with respect to the pre-activation values
        of the last layer. They are the starting point for the backprop algorithm. 
        '''

        output = self.calculateOutput(datapoint)
        oLayer = self.layers[-1]
        costDerivative = self.costFunction(output, datapoint.outputs, derivative=True)
        activationDerivative = oLayer.activationFunction(oLayer.preActs, derivative=True)
        return costDerivative * activationDerivative
    
    def _backprop(self, datapoint: Datapoint):
        endVals = self._getEndVals(datapoint)
        self.layers.reverse() # reverses order of the layers
        for i, layer in enumerate(self.layers):
            # update weight and bias gradients
            layer.weightsGradient -= np.matmul(layer.inputs.reshape(layer.numNodesIn, 1), np.atleast_2d(endVals))
            layer.biasGradient -= endVals.flatten()

            if i + 1 >= len(self.layers): # only triggers on the last layer, endvals don't need to be recalculated
                self.layers.reverse() # reverses back
                return

            # calculates the end values of the next layer, being the end values of the previous layer times the activation value
            # times the activation derivative
            endVals = np.matmul(layer.weightsArray, endVals.reshape(layer.numNodesOut, 1))
            endVals = endVals.reshape(1, layer.numNodesIn) * layer.activationFunction(self.layers[i + 1].preActs, derivative=True)

    def _updateValues(self, learnRate, batchSize):
        for layer in self.layers:
            layer.updateWeights(learnRate, batchSize)
            layer.updateBiases(learnRate, batchSize)
            
    def train(self, dataset, learnRate, epochs, batchSize = None, targetCost = 0, targetAcc = 1.1, printMode = False, showCostPlot = False, showAccPlot = False):

        '''
        trains the model for a specified amount of epochs, including options for setting a target cost or target accuracy
        which will stop the training after they are reached

        Arguments:
        dataset -> iterable of datapoint objects to train on, iterable of datapoint objects
        learnRate -> constant to adjust the rate of gradient descent. A good starting value is between 0.01 and 0.001, float
        epochs -> # of epochs to train over, int
        batchSize -> size of each batch, default is the same as the dataset size (no batching), int
        targetCost -> cost value which stops training at the end of the current epoch if reached, float
        targetAcc -> accuracy value which stops training at the end of the current epoch if reached (as a percentage, not count), float
        printMode -> prints the current epoch when it completes if True, bool
        showCostPlot -> shows a plot of cost vs. epoch at the end of training if True, bool
        showAccPlot -> shows a plot of accuracy vs. epoch at the end of training if True, bool

        Returns: None
        '''

        i = 0 # epoch counter
        cost = 1000 # arbitrarily high starting cost that will almost assuredly be above targetCost 
        acc = 0 # starting accuracy value that will definitely be below the target accuracy
        dataset = np.array(dataset) # makes sure the dataset is a numpy array
        datasetSize = len(dataset) # this value is used multiple times, so it is a variable as decreed by God

        if batchSize is None: # sets batchSize to the size of the full dataset (no batching) if not specified
            batchSize = datasetSize

        batchCount = int(np.ceil(datasetSize / batchSize))

        # pads the dataset so it works even if it can't batch cleanly (dataset size does not divide batch count)
        datasetPad = np.pad(dataset, (0, batchCount * batchSize - datasetSize)).reshape(batchCount, batchSize)

        costs = []
        accs = []

        while i < epochs and cost > targetCost and targetAcc > acc:
            for batch in datasetPad:
                for datapoint in batch:
                    if datapoint == 0: # breaks loop when it reaches padding
                        break
                    self._backprop(datapoint)
                
                self._updateValues(learnRate, batchSize) # updates values after each batch

            # these if statements exist to only calculate the costs and accuracies every epoch when necessary, to not bloat
            # the program with extra calculations when it's not
            if showCostPlot or targetCost > 0:
                cost = sum([self.cost(dp) for dp in dataset])
                costs.append(cost)

            if showAccPlot or targetAcc <= 1:
                acc = self.evaluate(dataset) / len(dataset)
                accs.append(acc)

            if printMode:
                print(f"epoch {i + 1} complete")

            i += 1
        
        if showCostPlot:
            plt.plot(np.arange(i), costs)
            plt.xlabel("epoch")
            plt.ylabel("total cost")
            plt.show()

        if showAccPlot:
            plt.plot(np.arange(i), accs)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.show()