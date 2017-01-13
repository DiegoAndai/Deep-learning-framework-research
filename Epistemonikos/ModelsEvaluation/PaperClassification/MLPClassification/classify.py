import tensorflow as tf
import numpy as np
import argparse
import os
import pickle
from functools import reduce
import datetime

def batcher(data, batch_size):
    index = 0
    not_flaged = True
    while index + batch_size <= len(data) and not_flaged:
        try:
            next_batch = data[index : index + batch_size]
        except IndexError:
            not_flaged = True
            raise StopIteration
        else:
            yield next_batch
            index += batch_size



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify papers.")
    parser.add_argument("--Train_test_path", help="Path to the pickled data.",
                        required=True)
    parser.add_argument("--Batch_size", help="Batch size for training.",
                        type = int, default = 10)

    args = parser.parse_args()
    tt_path = args.Train_test_path
    batch_size = args.Batch_size
    join_path = os.path.join


    with open(join_path(tt_path, "train_data"), "rb") as trd, \
            open(join_path(tt_path, "train_labels"), "rb") as trl, \
            open(join_path(tt_path, "test_data"), "rb") as ted, \
            open(join_path(tt_path, "test_labels"), "rb") as tel:
        train_data = pickle.load(trd)
        train_str_labels = pickle.load(trl)
        test_data = pickle.load(ted)
        test_str_labels = pickle.load(tel)

    train_hv_labels = list() #hot vector
    for str_label in train_str_labels:
        if str_label == "primary-study":
            train_hv_labels.append([1, 0])
        elif str_label == "systematic-review":
            train_hv_labels.append([0, 1])
        else:
            print("founded error in train data")

    test_hv_labels = list() #hot vector
    for str_label in test_str_labels:
        if str_label == "primary-study":
            test_hv_labels.append([1, 0])
        elif str_label == "systematic-review":
            test_hv_labels.append([0, 1])
        else:
            print("founded error in data")

    sess = tf.InteractiveSession()

    dimension = len(train_data[0])

    # Create the model
    x = tf.placeholder(tf.float32, [None, dimension])
    W = tf.Variable(tf.zeros([dimension, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    tf.global_variables_initializer().run()

    td_batcher = batcher(train_data, batch_size)
    tl_batcher = batcher(test_hv_labels, batch_size)

    # Train
    for i in range(1000):
        batch_xs = next(td_batcher)
        batch_ys = next(tl_batcher)
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

        current_p = precision.eval({x: test_data, y_: test_hv_labels})  # evaluate precision
        current_r = recall.eval({x: test_data, y_: test_hv_labels})  # evaluate recall

        avg_precision += current_p
        avg_recall += current_r

        print("Class {}:\nPrecision: {}\nRecall: {}".format(cls, current_p, current_r))

    avg_precision /= 2
    avg_recall /= 2
    print("Average precision: {}\nAverage recall: {}".format(avg_precision, avg_recall))
    print("Accuracy: {}\n".format(accuracy.eval({x: test_data, y_: test_hv_labels})))

    # Print confusion matrix
    yvsy_ = tf.transpose(tf.pack([tf.argmax(y, 1), tf.argmax(y_, 1)]))  # vector y packed with y_ and transposed

    confusion = np.zeros([2, 2], int)
    for p in yvsy_.eval(feed_dict={x: test_data, y_: test_hv_labels}):
        confusion[p[0], p[1]] += 1

    print(confusion)


    '''if len(test_labels) != len(predictions):
        print("dimensions error. labels: {}, predictions: {}".format(len(test_labels),
                                                                      len(predictions)))

    class_dimension = len(classes)
    conf_mtx = np.zeros([class_dimension, class_dimension])
    for i in range(0, len(predictions)):
        predicted_class = classes.index(predictions[i])
        actual_class = classes.index(test_labels[i])
        conf_mtx[actual_class][predicted_class] += 1
    np.set_printoptions(suppress=True)
    print(conf_mtx)

    #assert len(test_labels) == len(predictions)
    hits = 0
    for l, p in zip(test_labels, predictions):
        if l == p:
            hits += 1
    accuracy = hits / len(test_labels) #saved for output
    print(accuracy)

    recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0,class_dimension)))
    recall_sum = 0
    recall_list = []
    for i in range(0,class_dimension):
        rcl = recall(i)
        if not np.isnan(rcl):
            recall_sum += rcl
        recall_list.append((i, rcl))
        print('Recall {}: {:.5f}'.format(i, rcl))
    print()
    recall_mean = recall_sum/class_dimension
    print('Recall mean: {:.5f}'.format(recall_mean))

    precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0,class_dimension)))
    precision_sum = 0
    precision_list = list()
    for i in range(0,class_dimension):
        label_precision = precision(i)
        if not np.isnan(label_precision):
            precision_sum += label_precision
        precision_list.append((i, label_precision))
        print('Precision {}: {:.5f}'.format(i, label_precision))
    print()
    precision_mean = precision_sum/class_dimension
    print('Precision mean: {:.5f}'.format(precision_mean))

    output = ''
    output += 'Model: {}\n'.format(args.model_path)
    output += 'KNN classifier with k = {}\n'.format(args.K)
    output += 'span = {}\n'.format(args.span)
    output += 'Set: {}\n'.format(args.KNN_papers_set)
    output += 'Accuracy : {}\n'.format(accuracy)
    output += "RECALL\n"
    for rcl in recall_list:
        output += 'Recall {}: {:.5f}\n'.format(rcl[0], rcl[1])
    output += 'Recall mean: {:.5f}\n'.format(recall_mean)
    output += "PRECISION\n"
    for pcsn in precision_list:
        output += 'Precision {}: {:.5f}\n'.format(pcsn[0], pcsn[1])
    output += 'Precision mean: {:.5f}\n'.format(precision_mean)
    output += 'CONFUSSION MATRIX\n'
    output += str(conf_mtx)


    with open("output{}.txt".format(args.model_path.split('/')[-2]), "w") as out_file:
        out_file.write(output)

    #for testy, label in zip(test_data, test_labels):
    #    print(label, classifier.predict_proba(np.asarray([testy])), classifier.predict(np.asarray([testy])))'''
