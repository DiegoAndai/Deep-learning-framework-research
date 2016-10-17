from MNIST.mnist_data_loader import loader
import tensorflow as tf
import numpy as np

# Load data
datahandler = loader.MNIST('../mnist_data_loader')
datahandler.load_testing()
datahandler.load_training()

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

tf.initialize_all_variables().run()

# Train
for i in range(60000):
    batch_xs, batch_ys = datahandler.next_train_batch(10)
    train_step.run({x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted_classes = tf.argmax(y, 1)  # 1-D tensor with classes predicted for each image.
true_classes = tf.argmax(y_, 1)  # 1-D tensor with the actual classes.

avg_precision = avg_recall = 0
for cls in range(10):

    cls_is_predicted = tf.equal(predicted_classes, cls)  # True when cls was predicted, false otherwise.
    cls_is_not_predicted = tf.logical_not(cls_is_predicted)  # True when cls was not predicted.

    cls_is_label = tf.equal(true_classes, cls)  # True when actual value was cls.
    cls_is_not_label = tf.logical_not(cls_is_label)  # True when actual value was not cls.

    tp = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_label, cls_is_predicted), tf.float32))  # True positives.
    fp = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_predicted, cls_is_not_label), tf.float32))  # False positives.
    fn = tf.reduce_sum(tf.cast(tf.logical_and(cls_is_not_predicted, cls_is_label), tf.float32))  # False negatives.

    precision = tf.div(tp, tf.add(fp, tp))
    recall = tf.div(tp, tf.add(fn, tp))

    current_p = precision.eval({x: datahandler.np_test_images, y_: datahandler.np_one_hot_test_labels})  # evaluate precision
    current_r = recall.eval({x: datahandler.np_test_images, y_: datahandler.np_one_hot_test_labels})  # evaluate recall

    avg_precision += current_p
    avg_recall += current_r

    print("Class {}:\nPrecision: {}\nRecall: {}".format(cls, current_p, current_r))

avg_precision /= 10
avg_recall /= 10
print("Average precision: {}\nAverage recall: {}".format(avg_precision, avg_recall))
print("Accuracy: {}\n".format(accuracy.eval({x: datahandler.np_test_images, y_: datahandler.np_one_hot_test_labels})))

# Print confusion matrix
yvsy_ = tf.transpose(tf.pack([tf.argmax(y, 1), tf.argmax(y_, 1)]))  # vector y packed with y_ and transposed

confusion = np.zeros([10, 10], int)
for p in yvsy_.eval(feed_dict={x: datahandler.np_test_images, y_: datahandler.np_one_hot_test_labels}):
    confusion[p[0], p[1]] += 1

print(confusion)
