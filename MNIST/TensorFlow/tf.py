# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

tf.initialize_all_variables().run()

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
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

    current_p = precision.eval({x: mnist.test.images, y_: mnist.test.labels})  # evaluate precision
    current_r = recall.eval({x: mnist.test.images, y_: mnist.test.labels})  # evaluate recall

    avg_precision += current_p
    avg_recall += current_r

    print("Class {}:\nPrecision: {}\nRecall: {}\n".format(cls, current_p, current_r))

avg_precision /= 10
avg_recall /= 10
print("Average precision: {}\nAverage recall: {}".format(avg_precision, avg_recall))
print("Accuracy: {}\n".format(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
