#Evaluation of Keras, MXNet, TensorFlow and Torch through an MNIST implementation

We wondered, as beginners in the area of Machine Learning and Deep Learning, which were the main differences between the frameworks 
available in this area, but also which advantages and disadvantages these frameworks have depending on the context of development. So
in order to pacify our bewilderment, we took the time to implement the so called "Hello World!!!" of neural networks, the MNIST handwriten
digit classification, in four well known frameworks: Keras, MXNet, TensorFlow and Torch. You can see the detailed implementations in the folders in this directory. <br>
<br>
The neural network we chose to implement was overly simple: we flattened the 28&times;28 digit images to vectors (tensors) of 784 pixels, then we applied a linear layer and finally obtained a probability distribution for the possible classes (the digits) with softmax regression. The data that we fed to the implementation of this net in each framework was exactly the same, and it was obtained from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and processed by parsers that we found on the web and also edited ourselves. Finally, for each framework implementation, we calculated some metrics, accuracy, precission, recall, and a confussion matrix, to be precise. They did not vary greatly between implementations, which is nice.
<br>
<br>
The model was trained on a 60,000 digits training sample, and tested using 10,000 digits. The learning rate was setted to 0.01, using cross entropy as the loss function and gradient descent as the optimizer. Other parameters like momentum were not considered. The training process was extended to 10 epochs, with a batch size of 10.

##TensorFlow

This is a framework from Google designed for machine and deep learning. It works with data flow graphs, which represent mathematical 
operations and tensors and in all allow very fast computation, even using a language like Python, feature which comes in handy for big datasets and complicated neural nets. That is the main advantage of TensorFlow we recognised. An example of operation definition in TensorFlow is the creation of the neural network, defined [here](/MNIST/TensorFlow/tf.py) as the following: 
```python
x = tf.placeholder(tf.float32, [None, 784])  # input images, flattened. None means any amount of images
W = tf.Variable(tf.zeros([784, 10]))  # weight matrix
b = tf.Variable(tf.zeros([10]))  # bias vector
y = tf.nn.softmax(tf.matmul(x, W) + b)  # output of the network
```
After you define all of your operations, you have to execute them with run() or eval(). <br>
This performance boost, however, is attained by thinking slightly different from the usual, usually evidently sequential Pythonic programming, mainly because the moment in which an operation is executed depends on the graph, which is not literally visible and is made depending on the relationship between defined operations. Also, the implementation of a solution to a problem may vary greatly from a usual Python implementation of that solution because to avoid evaluating expressions with run() and eval() all the time, and therefore use TensorFlow the way it was made to be used, many operations (if not all) have to be tensor oriented, which partially eliminates the possibility of looping. An example of this problem arised for us when we wanted to obtain precission, recall and a confusion matrix. All we had were tensors with the labels and the predicted classes, and to access the individual elements of these tensors, eval() or run() would have had to be called many times, so we came up with a TensorFlow friendly solution (although it is not Pythonically intuitive) in the last part of [the implementation](/MNIST/TensorFlow/tf.py).

All in all ~~you're just another brick in the wall~~, TensorFlow is very useful an efficient, but it may take some time to adjust to the way it's meant to be used. So if you are looking for efficiency or you want to implement a well known network, TensorFlow is for you, but if you are a beginner and are not used to tensor operations or you want to meddle a lot with your variables in a pythonic way and don't care much about efficiency, maybe something else would be better.

Metrics:

| Metric | Value |
|:-------|:------|
| Accuracy | 0.92339 |
| Recall | 0.92241 |
| Precision | 0.92259 |

##Torch

This framework is not as efficient as TensorFlow, but has what TensorFlow lacks: flexibility. Basically, you can twist it in any direction you want. It also has different levels of abstraction, as specified in our framework comparison table. In our implementation of MNIST, we used a library called nn (neural networks), which is highly abstracted. A minor setback, although, might be that Lua, the programming language that Torch uses, has indexes that start at 1, not at 0 (fact that certainly gave us problems). Here is how we defined the network (not so different from the other frameworks):

```lua
-- nn definition

net = nn.Sequential()  -- sequential nn.

net:add(nn.Reshape(28*28))  -- flatten images. (could use View)
net:add(nn.Linear(28*28, #classes))  -- Fully connected layer.
net:add(nn.LogSoftMax())  -- Softmax layer.
```

Notice that, despite the part of [the code](/MNIST/Torch/torch.lua) for precission and recall consisting of more lines than the one of TensorFlow, it has a flow that is more natural for a programmer, consisting of loops and conditionals rather than of tensor operations (also the tensors are indexable!). Nevertheless, keep in mind that for the networks, tensor manipulation is most likely not avoidable, but probably abstractable.<br>
If you are into creating something new and/or you are not worried about extreme efficiency (because torch is still efficient) and you are looking for something that can bend easily to your will, then torch is your choice.  

Metrics:

| Metric | Value |
|:-------|:------|
| Accuracy | 0.8562 |
| Recall | 0.8553 |
| Precision | 0.8645 |


##Keras
This framework was developed by François Chollet, and it relies on either Theano or Tensorflow for it’s mathematical computation. It’s a high level library, focused on fast experimentation and easy prototyping. It has most of the options you would want to see on a neural network. It’s main advantage is how fast you can set up and use a model.

[Implementation](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/MNIST/Keras/Keras_mnist_mlp.ipynb)

Implementing the MNIST classifier net on Keras was, as expected, fast and straightforward, even though we were new to it. This is the principal objective of the framework, and it did wat it was build to do. The main sintaxis of Keras is very pythonic and intuitive, you declare the model, compile it and train/test it with raw numpy arrays. Metrics obtained were:

| Metric | Value |
|:-------|:------|
| Accuracy | 0.85360 |
| Recall | 0.84972 |
| Precision | 0.85565 |

The running time was approximately 40 seconds. Keras' training system provided live progress reports (see implementation) for every epoch, showing loss anc accuracy metrics as they develop. This is a good feature, expecially if you're training for a long time, as you can see if something isn't going well very soon, and don't have to wait untill the training ends. This framework doesn't yet have metrics like recall or precision. 

##Mxnet
Developed by a group of collaborators supported by companies like Intel, Nvidia and many more.  It focus on mixing symbolic and imperative programming in order to obtain both efficiency and flexibility. It supports over 7 programming languages, this is an important advantage over the other frameworks.

[Implementation](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/MNIST/Mxnet/Mxnet_mnist_mlp.ipynb)

Implementing the MNIST classifier net on Mxnet was challenging. The first steps with this framework were confusing, as it’s sintaxis and design isn’t intuitive. for example the separated declaration of layers and it's activations, which were connected later almost like distinct layers (see implementation). Maybe it’s principal characteristics and advantages like flexibility and the programming paradigm duality didn’t apply to a simple net like this one. Even though, results were good.  Metrics obtained were:

| Metric | Value |
|:-------|:------|
| Accuracy | 0.93720 |
| Recall | 0.93632 |
| Precision | 0.93861 |

Mxnet's training lasted 4 minutes approximately. That's longer than it's counterparts, but it's performance was better. Various metrics are implemented in this library, not recall nor precision, though. Accuracy was implemented, but it was calculated here manually with the confusion matrix as the other metrics, because when trying to use the implementation, a lot of errors came up, referring to the type of the input it needed. This kind of errors, of type and how to use tools from Mxnet were frequent, and as a beginner they cause distractions from the principal task, learning how to use the framework with a simple example.
<br>
##Conclusion

Implementing the net on these four different frameworks had us faced with our first neural network task. This is a factor to consider, as we never used tools like these before and were unexperienced, so our problems may not apply to more trained people, maybe they face other kind of difficulties. In this example, if we compare the four by:

**Beginner Difficulty**: Leaving experience in Python or Lua aside, Keras and Torch have a similar complexity when you’re starting to use them, being more beginner friendly than their counterparts.

**Intuitiveness**: Torch, Keras and Mxnet have similar syntax for this simple network. Making them equally intuitive. Whereas TensorFlow is less intuitive because it does not provide concrete layer definition, as we expected.

**Results**: Mxnet had slightly better metrics, but differences with TensorFlow were negligible. 

Finally, for this simple model and when starting to understand how neural networks are implemented, we recommend Keras if you’re more familiar with Python than Lua, and Torch otherwise. 
