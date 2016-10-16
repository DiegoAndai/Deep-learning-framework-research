#Evaluantion of Keras, MXNet, TensorFlow and Torch through an MNIST implementation

We wondered, as beginners in the area of Machine Learning and Deep Learning, which were the main differences between the frameworks 
available in this area, but also which advantages and disadvantages these frameworks have depending on the context of development. So
in order to pacify our bewilderment, we took the time to implement the so called "Hello World!!!" of neural networks, the MNIST handwriten
digit classification, in four well known frameworks: Keras, MXNet, TensorFlow and Torch. <br>
The neural network we chose to implement was overly simple: we flattened the 28&times;28 digit images to vectors (tensors) of 784 pixels, then we applied a linear layer and finally obtained a probability distribution for the possible classes (the digits) with softmax regression. The data that we fed to the implementation of this net in each framework was exactly the same, and it was obtained from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and processed by parsers that we found on the web and also edited ourselves.

##TensorFlow

This is a framework from Google designed for machine and deep learning. It works with data flow graphs, which represent mathematical 
operations and tensors and in all allow very fast computation, even using a language like Python, feature which comes in handy for big datasets and complicated neural nets. That is the main advantage of TensorFlow we recognised. An example of operation definition in TensorFlow is the creation of the neural network: 
```python
x = tf.placeholder(tf.float32, [None, 784])  # input images, flattened. None means any amount of images
W = tf.Variable(tf.zeros([784, 10]))  # weight matrix
b = tf.Variable(tf.zeros([10]))  # bias vector
y = tf.nn.softmax(tf.matmul(x, W) + b)  # output of the network
```
After you define all of your operations, you have to execute them with run() or eval(). <br>
This performance boost, however, is attained by thinking slightly different from the usual, usually evidently sequential Pythonic programming, mainly because the moment in which an operation is executed depends on the graph, which is not literally visible and is made depending on the relationship between defined operations. Also, the implementation of a solution to a problem may vary greatly from a usual Python implementation of that solution because to avoid evaluating expressions with run() and eval() all the time, and therefore use TensorFlow the way it was made to be used, many operations (if not all) have to be tensor oriented, which partially eliminates the possibility of looping. An example of this problem arised for us when we wanted to obtain precission, recall and a confusion matrix. All we had were tensors with the labels and the predicted classes, and to access the individual elements of these tensors, eval() or run() would have had to be called many times, so we came up with the following TensorFlow friendly solution (although is not Pythonically intuitive):
```python
# cls is the class for which we want precission and recall, true_classes and predicted_classes
# are the tensors with the labels and predictions, respectively.
cls_is_predicted = tf.equal(predicted_classes, cls)  # True when cls was predicted, false otherwise.
cls_is_not_predicted = tf.logical_not(cls_is_predicted)  # True when cls was not predicted.

cls_is_label = tf.equal(true_classes, cls)  # True when actual value was cls.
cls_is_not_label = tf.logical_not(cls_is_label)  # True when actual value was not cls.

tp = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_label, cls_is_predicted), tf.float32))  # True positives.
fp = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_predicted, cls_is_not_label), tf.float32))  # False positives.
fn = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_not_predicted, cls_is_label), tf.float32))  # False negatives.

precision = tf.div(tp, tf.add(fp, tp))
recall = tf.div(tp, tf.add(fn, tp))
```
All in all ~~you're just another brick in the wall~~, TensorFlow is very useful an efficient, but it may take some time to adjust to the way it's meant to be used. So if you are looking for efficiency or you want to implement a well known network, TensorFlow is for you, but if you are a beginner and are not used to tensor operations or you want to meddle a lot with your variables in a pythonic way and don't care much about efficiency, maybe something else would be better.

