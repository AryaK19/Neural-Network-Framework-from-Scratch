# Neural-Network-Framework-from-Scratch
 A framework for building neural networks and to get a grasp on how large frameworks like Pytorch and Tensorflow work.

# Getting Started
To get started with NeuroFlow, follow these steps:

1.Copy the NeuroFlow.py file into your project directory.

2.Import the Sequential class from NeuroFlow: from NeuroFlow import Sequential.
#  
(Before you start with this framework I highly recommend to have a look in 'NN.ipynb' which is a same model, but has only one hidden layer hard coded. This will help you better grasp the code in the actual model)
#  
You can now initialize the model (prefer the two examples in 'MNIST.ipynb' or 'DIABETES.ipynb'. Where I have implemented an neural network for image data and simple data.

While initializing the model 2 attributes are necessary i.e. input size and output size, after which you can add Dense layers in the network with a specified number as per the example

For the fit method in Sequential, verboseInterval defines how frequently the loss is printed and stored in 'history'(loss).

Finally, you can save the model by save method and load the model from NeuroFlow.loadModel() or simply - from NeuroFlow import loadModel

Thanks and keep learning !!



#   
Still working on it (adding more features and layers)!  , ALL THE UPDATES ARE IN FOLDER 'Updates' :)
#  
