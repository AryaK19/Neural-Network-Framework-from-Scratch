import numpy as np
import pickle

class Sequential:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.filters = []
        self.layerNo = 0
        self.layers = []
        self.activations = []
        self.compiled = False
        self.inputShape = []
        self.outputShape = []
        self.startDense = 0

    def compile(self):
        if self.layerNo < 2:
            print("The Number of Layers is less than 2")
            return
        self.compiled = True

    def setInputShape(self, inputShape):
        self.inputShape = inputShape

    def addLayerConv2D(self, filterNo: int, filterSize: tuple, activation: str, inputShape: tuple = None):
        if self.layerNo == 0 and inputShape is not None:
            self.inputShape.append(inputShape)
        if self.layerNo == 0 and inputShape is None:
            raise ValueError("Input shape must be defined for the first layer")

        self.filters.append(np.random.randn(filterNo, filterSize[0], filterSize[1]) / (filterSize[0] * filterSize[1]))

        self.layers.append("Conv2D")
        self.activations.append(activation)

        if self.layerNo == 0:
            inputHeight, inputWidth = inputShape
        else:
            inputHeight, inputWidth = self.outputShape[-1][1:]

        outputHeight = inputHeight - filterSize[0] + 1
        outputWidth = inputWidth - filterSize[1] + 1
        self.outputShape.append((filterNo, outputHeight, outputWidth))

        self.layerNo += 1
        self.startDense += 1

    def addLayerMaxPool(self, filterSize: tuple):
        self.maxFilterSize = filterSize
        self.activations.append("")
        self.inputShape.append(self.outputShape[-1])
        outputHeight = self.inputShape[-1][1] // filterSize[0]
        outputWidth = self.inputShape[-1][2] // filterSize[1]
        self.outputShape.append((self.inputShape[-1][0], outputHeight, outputWidth))
        self.layers.append("MaxPool")
        self.layerNo += 1
        self.startDense += 1

    def addLayerFlatten(self):
        self.inputShape.append(self.outputShape[-1])
        flattenedSize = np.prod(self.inputShape[-1])
        self.outputShape.append((flattenedSize,))
        self.activations.append("")
        self.layers.append("Flatten")
        self.layerNo += 1
        self.startDense += 1

    def addLayerDense(self, size: int, activation: str):
        if self.layerNo == 0:
            self.outputShape.append((size,))
            self.inputShape.append(None)
            self.layerNo += 1
            self.activations.append("")
            return

        self.weights.append(np.random.randn(self.outputShape[-1][0], size) * np.sqrt(2. / self.outputShape[-1][0]))
        self.bias.append(np.zeros((1, size)))
        self.activations.append(activation)
        self.layerNo += 1
        self.inputShape.append(self.outputShape[-1])
        self.outputShape.append((size,))
        self.layers.append("Dense")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def reluDerivative(self, x):
        return np.where(x <= 0, 0, 1)

    def softmax(self, x):
        expX = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expX / np.sum(expX, axis=1, keepdims=True)

    def computeLoss(self, yTrue, yPred):
        epsilon = 1e-8
        yPred = np.clip(yPred, epsilon, 1. - epsilon)
        m = yTrue.shape[0]
        loss = -1 / m * np.sum(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))
        return loss


    def forwardConv2D(self, images, activation, layerNo):
        batchSize = images.shape[0]
        filterCount, filterHeight, filterWidth = self.filters[layerNo].shape
        imageHeight, imageWidth = images.shape[1:3]
        convHeight = imageHeight - filterHeight + 1
        convWidth = imageWidth - filterWidth + 1

        conv2D = np.zeros((batchSize, filterCount, convHeight, convWidth))

        for n in range(batchSize):
            for f in range(filterCount):
                for i in range(convHeight):
                    for j in range(convWidth):
                        conv2D[n, f, i, j] = np.sum(
                            images[n, i:i + filterHeight, j:j + filterWidth] * self.filters[layerNo][f]
                        )

        conv2D = self.relu(conv2D)
        return conv2D

    def forwardMaxPool(self, images, activation=None, layerNo=None):
        batchSize, channels, height, width = images.shape
        poolHeight, poolWidth = self.maxFilterSize
        outHeight = height // poolHeight
        outWidth = width // poolWidth

        maxp = np.zeros((batchSize, channels, outHeight, outWidth))

        for n in range(batchSize):
            for c in range(channels):
                for i in range(outHeight):
                    for j in range(outWidth):
                        maxp[n, c, i, j] = np.amax(
                            images[n, c, i * poolHeight:(i + 1) * poolHeight, j * poolWidth:(j + 1) * poolWidth]
                        )
        return maxp

    def forwardFlatten(self, images, activation=None, layerNo=None):
        batchSize = images.shape[0]
        return images.reshape(batchSize, -1)

    def forwardDense(self, data, activation, layerNo):
        z = np.dot(data, self.weights[layerNo - self.startDense]) + self.bias[layerNo - self.startDense]
        output = eval(f"self.{activation}(z)")
        return output
    
    def backwardDense(self, X, y, learningRate=0.01):
        m = X.shape[0]

        
        dw = [0] * len(self.weights)
        db = [0] * len(self.bias)

        
        dz = self.output[-1] - y
        
        for i in range(self.layerNo - 1, self.startDense - 1, -1):
            
            dw[i - self.startDense] = (1 / m) * np.dot(self.output[i - 1].T, dz)
            db[i - self.startDense] = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            
            if i > self.startDense:
                dz = np.dot(dz, self.weights[i - self.startDense].T) * eval(f"self.{self.activations[i - 1]}Derivative(self.output[i - 1])")

        for i in range(len(self.weights)):
            self.weights[i] -= learningRate * dw[i]
            self.bias[i] -= learningRate * db[i] 


    def backwardFlatten(self, dA, layerNo):
        originalShape = self.inputShape[layerNo]
        return dA.reshape(originalShape)
    

    def backwardMaxPool(self, dA, images, layerNo):
        batchSize, channels, height, width = images.shape
        poolHeight, poolWidth = self.maxFilterSize
        outHeight = height // poolHeight
        outWidth = width // poolWidth

        dMaxPool = np.zeros_like(images)

        for n in range(batchSize):
            for c in range(channels):
                for i in range(outHeight):
                    for j in range(outWidth):
                        hStart = i * poolHeight
                        hEnd = hStart + poolHeight
                        wStart = j * poolWidth
                        wEnd = wStart + poolWidth
                        poolRegion = images[n, c, hStart:hEnd, wStart:wEnd]
                        maxVal = np.max(poolRegion)
                        mask = (poolRegion == maxVal)
                        dMaxPool[n, c, hStart:hEnd, wStart:wEnd] += dA[n, c, i, j] * mask
        return dMaxPool

    def backwardConv2D(self, dZ, images, layerNo):
        batchSize, filterCount, convHeight, convWidth = dZ.shape
        filterCount, filterHeight, filterWidth = self.filters[layerNo].shape
        batchSize, channels, imageHeight, imageWidth = images.shape

        dFilters = np.zeros_like(self.filters[layerNo])
        dImages = np.zeros_like(images)

        for n in range(batchSize):
            for f in range(filterCount):
                for i in range(convHeight):
                    for j in range(convWidth):
                        dFilters[f] += dZ[n, f, i, j] * images[n, :, i:i + filterHeight, j:j + filterWidth]
                        dImages[n, :, i:i + filterHeight, j:j + filterWidth] += dZ[n, f, i, j] * self.filters[layerNo][f]

        return dFilters, dImages
    

    def backProp(self, X, y, learningRate=0.01):
        m = X.shape[0]

        dZ = self.output[-1] - y  

        for i in range(self.layerNo - 1, -1, -1):
            layer = self.layers[i]

            if layer == "Dense":
                dZ = self.backwardDense(self.output[i], dZ, learningRate)

            elif layer == "Flatten":
                dZ = self.backwardFlatten(dZ, i)

            elif layer == "MaxPool":
                dZ = self.backwardMaxPool(dZ, self.output[i], i)

            elif layer == "Conv2D":
                dFilters, dImages = self.backwardConv2D(dZ, self.output[i], i)
                self.filters[i] -= learningRate * dFilters
                dZ = dImages

        return dZ

    def predict(self, X):  
        self.output = X
        for i, layer in enumerate(self.layers):
            self.output = eval(f"self.forward{layer}(self.output, activation='{self.activations[i]}', layerNo={i})")
            
        return self.output

    def fit(self, X, y, epochs=10, batch_size=16, learningRate=0.001, verbose=True):
        if not self.compiled:
            print("Model is not compiled")
            return

        loss = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                batch_loss = 0.0
                for j in range(batch_size):
                    self.output = self.predict(X_batch[j].reshape(1, *X_batch[j].shape))
                    current_loss = self.computeLoss(y_batch[j].reshape(1, *y_batch[j].shape), self.output[-1])
                    batch_loss += current_loss
                    
                    self.backProp(X_batch[j].reshape(1, *X_batch[j].shape), y_batch[j].reshape(1, *y_batch[j].shape), learningRate)

                epoch_loss += batch_loss / batch_size

            loss.append(epoch_loss / X.shape[0])

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss[-1]}")

        if verbose:
            print("Training Complete")

        return np.array(loss)

    def summary(self):
        if self.layerNo == 0:
            print("Model has no layers.")
            return
        
        print("")
        print("=================")
        print("| Model Summary |")
        print("=================")
        totalParams = 0
        
        for i, (layer_type, activation) in enumerate(zip(self.layers, self.activations)):
            layerNum = i 
            inputShape = self.inputShape[i ]
            outputShape = self.outputShape[i ]
            
            print("")
            print("---------------------------------------------------------------")
            print(f"Layer {layerNum} ({layer_type}):")
            print("---------------------------------------------------------------")
            print(f"  Input Shape: {inputShape}\t\t|\tOutput Shape: {outputShape}")
            
            if layer_type == "Conv2D":
                w = self.filters[i]
                b = None  
            elif layer_type == "Dense":
                w = self.weights[i - self.startDense]
                b = self.bias[i - self.startDense]
            
            if w is not None:
                layerParams = w.size + (b.size if b is not None else 0)
                print(f"  Weights Shape: {w.shape}\t|\tBiases Shape: {b.shape if b is not None else None}")
                print(f"  Activation: {activation}")
                print(f"  Number of parameters: {layerParams}")
                totalParams += layerParams
            else:
                print(f"  Activation: {activation}")
                print(f"  Number of parameters: 0")
            
            print("================================================================")
        
        print("")
        print("----------------------------------------------------------------")
        print(f"Output Layer {self.layerNo + 1}:")
        print("----------------------------------------------------------------")
        print(f"  Input Dim: {self.outputShape[-1][0]}")
        print("================================================================\n")
        print("=============================================================")
        print(f"|  Total number of parameters: {totalParams}  |")
        print("=============================================================")

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def loadModel(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
        return file
