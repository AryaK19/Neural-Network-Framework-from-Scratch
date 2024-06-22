import numpy as np
import pickle


class Sequential:
    def __init__(self, inputSize,outputSize,outputLayerActivation='sigmoid'):

        self.weights = [np.random.randn(inputSize, outputSize) * np.sqrt(2. / inputSize)]
        self.bias = [np.zeros((1, outputSize))]
        self.layers = 0
        self.activations = [outputLayerActivation]

    def addLayerDense(self,size,activation='relu'):
        self.layers +=1
        self.weights.append(np.random.randn(size, self.weights[-1].shape[1]) * np.sqrt(2. / size))
        self.weights[-2] = np.random.randn(self.weights[-2].shape[0], self.weights[-1].shape[0]) * np.sqrt(2. / self.weights[-1].shape[0])

        self.bias.append(np.zeros((1,self.weights[-1].shape[1])))
        self.bias[-2] = np.zeros((1,size))        
        
        self.activations.insert(-2,activation)

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def computeLoss(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        m = y_true.shape[0]
        loss = -1 / m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss


    def predict(self, X):
        self.output = [X]
       
        for i in range(self.layers+1):
            self.z = np.dot(self.output[-1],self.weights[i]) + self.bias[i]
            self.output.append(eval(f"self.{self.activations[i]}(self.z)"))
            
        return self.output[-1]
    
    def backward(self, X, y, learningRate=0.01):
        m = X.shape[0]

        dz = [self.output[-1] - y]
        dw = [(1 / m) * np.dot(self.output[-2].T, dz[0])]
        db = [(1 / m) * np.sum(dz[0], axis=0, keepdims=True)]

        for i in range(self.layers,0,-1):
            dz = [np.dot(dz[0], self.weights[i].T) * eval(f"self.{self.activations[i-1]}_derivative(self.output[i])")] + dz
            dw = [(1 / m) * np.dot(self.output[i-1].T, dz[0])] + dw
            db = [(1 / m) * np.sum(dz[0], axis=0, keepdims=True)] + db

        for i in range(self.layers+1):
            
            self.weights[i] -= learningRate * dw[i]
            self.bias[i] -= learningRate * db[i]
            

    def fit(self,X,y,epochs,learningRate=0.01,verbose=True,verboseInterval=100):
        loss = []
    
        for epoch in range(epochs):
            pred = self.predict(X)
            self.backward(X, y,learningRate)

            if epoch%verboseInterval == 0 and verbose:
                print(f"\nEpoch: {epoch}\n|",end='')
                print("====================",end='')
                loss.append(self.computeLoss(y,pred))

            if epoch%verboseInterval == 0 and verbose:
                print(f"|   Loss: {loss[-1]}")

        if verbose:
            print("\nTraining Complete")
            
        return np.array(loss)
    
    def summary(self):
        print("")
        print("=================")
        print("| Model Summary |")
        print("=================")
        totalParams = 0
        for i, (w, b, activation) in enumerate(zip(self.weights, self.bias, self.activations)):
            layer_num = i + 1
            inputDim = w.shape[0]
            ouputDim = w.shape[1]
            layerParams = w.size + b.size
            print("")
            print("------------------------------------------------------------")
            print(f"Layer {layer_num}:")
            print("------------------------------------------------------------")
            print(f"  Input Dim: {inputDim}\t\t\t|\tOutput Dim: {ouputDim}")
            print(f"  Weights Shape: {w.shape}\t\t|\tBiases Shape: {b.shape}") 
            print(f"  Activation: {activation}")
            print(f"  Number of parameters: {layerParams}")
            print("=============================================================")
            totalParams += layerParams
        print(f"|                Total number of parameters: {totalParams}             |")
        print("=============================================================")


    def save(self,filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def loadModel(filename):
    with open(filename,'rb') as f:
        file = pickle.load(f)

        return file
    

