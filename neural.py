import numpy as np # import

# Jack Eckert - 4/04/2025

# ACTIVATION FUNCTIONS -----------------------------------------------------------------------------------------------
def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / (1 + np.exp(-x)) ** 2
    else:
        return 1 / (1 + np.exp(-x))
    
def ReLu(x, derivative=False):
    if derivative:
        return int(x > 0)
    else:
        return np.max(x, 0)

# COST FUNCTIONS -----------------------------------------------------------------------------------------------------
def square(x, derivative=False):
    if derivative:
        return 2 * x
    else:
        return x ** 2
    
# MISC ---------------------------------------------------------------------------------------------------------------
def formatData(array, seperator: int):
    lst = []
    for row in array:
        lst.append(Datapoint(row[:seperator], row[seperator:]))
    return np.array(lst)

# CLASSES  -----------------------------------------------------------------------------------------------------------
class Datapoint():
    def __init__(self, input, output):
        self.inputs = np.array(input)
        self.outputs = np.array(output)

class Layer():
    def __init__(self, numNodesIn: int, numNodesOut: int, activationFunction, costFunction, weightsRange = (-5, 5), biasRange = (-0.1, 0.1)):
        wL = weightsRange[0]
        wH = weightsRange[1]
        bL = biasRange[0]
        bH = biasRange[1]
    
        self.biasArray = np.random.random(size=numNodesOut) * (bH - bL) + bL
        self.weightsArray = np.random.random(size=(numNodesIn, numNodesOut)) * (wH - wL) + wL

        self.numNodesOut = numNodesOut
        self.numNodesIn = numNodesIn
        self.activationFunction = activationFunction
        self.costFunction = costFunction
        self.preActs = None
        self.outputs = None
        self.inputs = None
        self.biasGradient = np.zeros(shape=(numNodesOut))
        self.weightsGradient = np.zeros(shape=(numNodesIn, numNodesOut))

    def calculateOutputs(self, inputs):
        self.inputs = np.array(inputs)
        self.preActs = (np.matmul(self.inputs, self.weightsArray) + self.biasArray)
        self.outputs = np.apply_along_axis(self.activationFunction, 0, self.preActs)
        return np.apply_along_axis(self.activationFunction, 0, self.preActs)
    
    def setWeights(self, newWeights):
        self.weightsArray = newWeights

    def setBiases(self, newBiases):
        self.biasArray = newBiases

    def layerCost(self, expectedOutput, actualOutput):
        return np.apply_along_axis(self.costFunction, 0, (actualOutput - expectedOutput))
    
    def updateWeights(self):
        self.weightsArray += self.weightsGradient

    def updateBiases(self):
        self.biasArray += self.biasGradient
    
class NeuralNetwork():
    def __init__(self, shape: tuple, activationFunction = sigmoid, costFunction = square):
        self.layers = []
        for i in range(len(shape) - 1):
            self.layers.append(Layer(shape[i], shape[i + 1], activationFunction, costFunction))
        self.size = len(self.layers) + 1
        self.costFunction = costFunction
 
    def calculateOutput(self, inputs):
        put = inputs
        for layer in self.layers:
            put = layer.calculateOutputs(put)
        return put
    
    def Cost(self, dataPoint = Datapoint, returnTotal=True):
        output = self.calculateOutput(dataPoint.inputs)
        oLayer = self.layers[-1]

        if returnTotal:
            return np.sum(oLayer.layerCost(dataPoint.outputs, output))
        else:
            return oLayer.layerCost(dataPoint.outputs, output)
        
    def getEndVals(self, inputs):
        output = self.calculateOutput(inputs)
        oLayer = self.layers[-1]
        costDerivative = np.apply_along_axis(self.costFunction, 0, output, True)
        activationDerivative = np.apply_along_axis(oLayer.activationFunction, 0, oLayer.preActs, True)
        return costDerivative * activationDerivative
    
    def backprop(self, inputs):
        endVals = self.getEndVals(inputs)
        self.layers.reverse() # reverses order of the layers
        for i, layer in enumerate(self.layers):
            # update weight and bias gradients
            layer.weightsGradient = -(np.matmul(np.reshape(layer.inputs.reshape(layer.numNodesIn, 1)), np.atleast_2d(endVals)))
            layer.biasGradient = -endVals

            if i + 1 >= len(self.layers):
                self.layers.reverse()
                return

            endVals = np.matmul(endVals, None)
        
    def learn(self, inputs):
        pass
            
    def train(self, dataset):
        pass

# TESTING
if __name__ == '__main__':
    XORarr = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    XORarr = formatData(XORarr, 2)

    testN = NeuralNetwork((2, 2, 1))

    d1 = Datapoint([0, 1], [1])

    print(testN.calculateOutput(d1.inputs))
    print(testN.Cost(d1))