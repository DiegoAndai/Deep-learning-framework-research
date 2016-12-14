"""This code is based on a TensorFlow tutorial about Word2Vec at
https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html"""


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

n = input("Enter 'original' to train with Tensorflow's tutorial words, 'wiki' to use wikipedia corpus, enter 'w' to load Epistemonikos data from words "
          "file or just enter to parse Epistemonikos data from json file ").lower()

if os.path.exists('../words.txt') and n == "w":
    with open('../words.txt') as w_file:
        words = [word.rstrip() for word in w_file]
elif not n:
<<<<<<< HEAD
    with open("../documents_array.json", "r") as json_file:
=======
    with open("documents_array.json", "r") as json_file:
>>>>>>> 7b47a33df15ac0516d698b3ae503d3c1a14b0ecd
        loaded = json.load(json_file)

    reader = PaperReader(loaded)
    print("generating words")
    reader.generate_words_list()
    reader.save_words()
    words = reader.words
elif n == "wiki":
    keep = string.ascii_lowercase + '-'
    with open("allwiki", "r") as allwiki:
        words = []
        for line in allwiki:
            line = line.lower()
            line_words = line.split()
            for word in line_words:
                valid = True
                for ch in "123456789</>":
                    if ch in word:
                        valid = False
                if valid:
                    words.append("".join(ch for ch in word if ch in keep))
elif n == "original":
    # Step 1: Download the data.
    url = 'http://mattmahoney.net/dc/'


    def maybe_download(filename, expected_bytes):
      """Download a file if not present, and make sure it's the right size."""
      if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
      else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename

    filename = maybe_download('text8.zip', 31344016)


    # Read the data into a list of strings.
    def read_data(filename):
      """Extract the first file enclosed in a zip file as a list of words"""
      with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
      return data

    words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
print("step 2")
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]  # sub_list ['UNK', -1] is used to count uncommon words.
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # List of tuples of words with
  # their appearances. The parameter of most_common determines how many most common words it will return.
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)  # create a dictionary of the most common words with unique integers as values.
    # Notice that "UNK" will have id = 0.
  data = list()  # original list of words, but with id's instead of words.
  unk_count = 0  # appearances of uncommon words
  for word in words:
    if word in dictionary:  # if it is common enough
      index = dictionary[word]  # get it's id
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count  # update uncommon appearances.
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # to get the word by its id.
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
print("step 3")
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
print("step 4")


batch_size = 50
embedding_size = 1200  # Dimension of the embedding vector.
skip_window = 3      # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 10000 # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

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

# Step 5: Begin training.
print("step 5")

num_steps = 300001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  loss_progress = list()
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

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
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]  # note argsort gives indices, not elements.
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)

  final_embeddings = normalized_embeddings.eval()

with open("embedding", "wb") as embed_file:
    pickle.dump(final_embeddings, embed_file)

with open("count", "wb") as count_file:
    pickle.dump(count, count_file)

with open("reverse_dictionary", "wb") as reverse_dictionary_file:
    pickle.dump(reverse_dictionary, reverse_dictionary_file)

#graph of loss progress
import matplotlib.pyplot as plt
from numpy import arange
range_ = arange(0, len(loss_progress), 1)
plt.plot(range_, loss_progress)
#plt.savefig("Adamoptimizer")
plt.show()
