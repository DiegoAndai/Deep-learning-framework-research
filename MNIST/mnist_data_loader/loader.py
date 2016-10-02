import os
import struct
import numpy as np
from array import array


class MNIST(object):
    def __init__(self, path=''):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

        # Following attributes by Vicente Valencia
        self.epoch_index = 0
        self.np_train_images = None
        self.np_test_images = None
        self.np_one_hot_train_labels = None
        self.np_one_hot_test_labels = None

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        # Following 2 lines by Vince-Valence
        self.np_test_images = np.multiply(np.array(ims), 1.0 / 255.0)
        self.np_one_hot_test_labels = self.dense_to_one_hot(np.array(labels), 10)

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        # Following 2 lines by Vicente Valencia
        self.np_train_images = np.multiply(np.array(ims), 1.0 / 255.0)
        self.np_one_hot_train_labels = self.dense_to_one_hot(np.array(labels), 10)

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=160):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):

        """Convert class labels from scalars to one-hot vectors.
        Function courtesy of TensorFlow."""

        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    # Following code by Vicente Valencia

    def next_train_batch(self, batch_size):

        """Returns two numpy arrays (images and labels)
        of size batch_size from the training set. Function
        strongly based on TensorFlow function with similar name."""

        start = self.epoch_index
        self.epoch_index += batch_size
        if self.epoch_index > len(self.np_train_images):
            start = 0
            self.epoch_index = batch_size
            # Shuffle the data set
            permutation = np.arange(len(self.np_train_images))
            np.random.shuffle(permutation)
            self.np_train_images = self.np_train_images[permutation]
            self.np_one_hot_train_labels = self.np_one_hot_train_labels[permutation]
        end = self.epoch_index

        return self.np_train_images[start: end], self.np_one_hot_train_labels[start: end]
