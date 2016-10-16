#Evaluantion of Keras, MXNet, TensorFlow and Torch through an MNIST implementation

We wondered, as beginners in the area of Machine Learning and Deep Learning, which were the main differences between the frameworks 
available in this area, but also which advantages and disadvantages these frameworks have depending on the context of development. So
in order to pacify our bewilderment, we took the time to implement the so called "Hello World!!!" of neural networks, the MNIST handwriten
digit classification, in four well known frameworks: Keras, MXNet, TensorFlow and Torch. <br>
The neural network we chose to implement was overly simple: we flattened the 28&times;28 digit images to vectors (tensors) of 784 pixels, then we applied a linear layer and finally obtained a probability distribution for the possible classes (the digits) with softmax regression. The data that we fed to the implementation of this net in each framework was exactly the same, and it was obtained from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and processed by parsers that we found on the web and also edited ourselves.

##TensorFlow

This is a framework from Google designed for machine and deep learning. It works with data flow graphs, which represent mathematical 
operations and tensors and in all allow very fast computation, even using a language like Python, which comes in handy for big datasets
and complicated neural nets. That is the main advantage of TensorFlow. 
