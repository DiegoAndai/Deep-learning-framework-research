# This code is based on a TensorFlow tutorial about Word2Vec at
# https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html
# and it has been modified


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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from open_documents import PaperReader
import collections
import math
import os
import random
import zipfile
import pickle
import string

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

with open("wiki_data", "rb") as _file:
    wiki_data = pickle.load(_file)

with open("episte_data", "rb") as _file:
    episte_data = pickle.load(_file)

with open("reverse_dictionary", "rb") as _file:
    reverse_dictionary = pickle.load(_file)


# Step 3: Function to generate a training batch for the skip-gram model.
print("step 3")


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

data_index = 0
wiki_batch, wiki_labels = generate_batch(wiki_data, batch_size=4, num_skips=2, skip_window=1)
data_index = 0
episte_batch, episte_labels = generate_batch(episte_data, batch_size=4, num_skips=2, skip_window=1)

for i in range(4):
    print("wiki")
    print(wiki_batch[i], reverse_dictionary[wiki_batch[i]],
          '->', wiki_labels[i, 0], reverse_dictionary[wiki_labels[i, 0]])
for i in range(4):
    print("episte")
    print(episte_batch[i], reverse_dictionary[episte_batch[i]],
          '->', episte_labels[i, 0], reverse_dictionary[episte_labels[i, 0]])

# Step 4: Build and train a skip-gram model.
print("step 4")

vocabulary_size = 50000
batch_size = 50
embedding_size = 1200  # Dimension of the embedding vector.
skip_window = 4  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 10000  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

def define_graph():

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(np.array(valid_examples), dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                           num_sampled, vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 0.1.
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)  # matrix (16x50000) in which each line has
        # the similarity values

        # Add variable initializer.
        init = tf.initialize_all_variables()
    return graph, optimizer, loss, init, similarity, train_inputs, train_labels, normalized_embeddings

# Step 5: Begin training.
print("step 5")

num_steps = 200001

def train_and_plot(data, data_name = None):
    if data_name:
        print(data_name)

    graph, optimizer, loss, init, similarity, train_inputs, train_labels, normalized_embeddings = define_graph()
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        average_loss = 0
        loss_progress = list()
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                loss_progress.append(average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 100000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # note argsort gives indices, not elements.
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()

    if data_name:
        with open(data_name + "embedding", "wb") as embed_file:
            pickle.dump(final_embeddings, embed_file)


    # graph of loss progress
    import matplotlib.pyplot as plt
    from numpy import arange

    range_ = arange(0, len(loss_progress), 1)
    plt.plot(range_, loss_progress)
    # plt.savefig("Adamoptimizer")
    plt.show()

train_and_plot(wiki_data, "wiki")
train_and_plot(episte_data, "episte")
