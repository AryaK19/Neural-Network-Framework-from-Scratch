# Neural-Network-Framework-from-Scratch
 A framework for building neural networks and to get a grasp on how large frameworks like Pytorch and Tensorflow work.


For how to run :
To get started, you can simply copy the 'NeuroFlow.py' file in any of your projects and by using - from NeuroFlow import Sequential.
You can now initialize the model (prefer the two examples in 'MNIST.ipynb' or 'DIABETES.ipynb'. Where I have implemented an neural network for image data and simple data.

While initializing the model 2 attributes are necessary i.e. input size and output size, after which you can add Dense layers in the network with a specified number as per the example

For the fit method in Sequential, verboseInterval defines how frequently the loss is printed and stored in 'history'(loss).

Finally, you can save the model by save method and load the model from NeuroFlow.loadModel() or simply - from NeuroFlow import loadModel

Thanks and keep learning !!
