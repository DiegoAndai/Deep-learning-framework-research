# TODO: make compatible with cmd_paper_processing.py output
# TODO: add cmd line option for the words to consider from each abstract and for the number of nearest neighbours.

from sklearn.neighbors import KNeighborsClassifier
from open_documents import PaperReader

import numpy as np
import tensorflow as tf
import argparse
import pickle
import json
import collections
import os



class DocumentSpace:

    class_sess = tf.Session()

    def __init__(self, language_model, lang_mod_order, span):
        self.vectors_by_type = list() #structure: [(type, vector)]
        self.span = span
        self.lang_mod_order = lang_mod_order # list of words of the language model (to know which
        # word an embedding represents)

        with DocumentSpace.class_sess.as_default():
            self.language_model = tf.nn.l2_normalize(language_model,
                                                     1).eval()  # Word embeddings used to get vectors from text.

    def get_abs_vectors(self, to_classify):

        """ Generates a vector for every abstract to be classified.
        :parameter to_classify: JSON array (python list) containing
        the abstracts (dicts) of the papers to be classified.
              """

        labels = [paper["classification"] for paper in to_classify]  # ground truth

        print("Calculating abstracts' vectors...")
        reader = PaperReader(to_classify)
        abs_vectors = list()
        parsed = 0
        n_abstracts = len(reader)
        print("started vector build")
        for abstract in reader:
            abs_count = collections.Counter(abstract[:self.span])
            vector = []
            for word in self.lang_mod_order:
                if word in abs_count:
                    vector.append(abs_count[word])
                    del abs_count[word]
                else:
                    vector.append(0)
            freq = np.array(vector, ndmin=2)
            nwords = sum(vector)
            rel_freq = freq / nwords if nwords else freq
            abs_vector = np.dot(rel_freq, self.language_model).squeeze()
            abs_vectors.append(abs_vector)
            parsed += 1
            if not parsed % 1000:
                print("{}/{}".format(parsed, n_abstracts))
        assert n_abstracts == parsed
        assert len(abs_vectors) == n_abstracts
        self.vectors_by_type = list(zip(labels, abs_vectors))
        print("Finished calculating abstracts' vectors.")

    def get_slices(self, train_size, test_size):
        train = self.vectors_by_type[:train_size]
        train_data = [t[1] for t in train]
        train_labels = [t[0] for t in train]


        test = self.vectors_by_type[train_size: train_size + test_size]
        test_data = [t[1] for t in test]
        test_labels = [t[0] for t in test]

        return train_data, train_labels, test_data, test_labels


if __name__ == "__main__":

    if os.path.exists("test_data"):
        print("opening previous data")
        with open("train_data", "rb") as trd, \
             open("train_labels", "rb") as trl, \
             open("test_data", "rb") as ted, \
             open("test_labels", "rb") as tel:
            train_data = pickle.load(trd)
            train_labels = pickle.load(trl)
            test_data = pickle.load(ted)
            test_labels = pickle.load(tel)
    else:
        parser = argparse.ArgumentParser(description="Classify papers.")
        parser.add_argument("--model_path", help="Path to a file containing pickled embeddings as a numpy array.",
                            required=True)
        parser.add_argument("--words_path", help="Path to a pickled list of the words in the model, "
                                                 "in the order of the embeddings.",
                            required=True)
        parser.add_argument("--span", help="Number of words used for classification, counting from"
                                                 "the start of the abstract", type=int, default=10)
        parser.add_argument("--papers_path", help="Path to json array of papers to classify.",
                            required=True)
        args = parser.parse_args()

        with open(args.model_path, 'rb') as model_file, \
             open(args.words_path, 'rb') as words_file:

            model = pickle.load(model_file)
            model_order = pickle.load(words_file)

        with open(args.papers_path, 'r') as json_file:
            papers = json.load(json_file)


        Space = DocumentSpace(model, model_order, args.span)
        Space.get_abs_vectors(papers[:105000])
        train_data, train_labels, test_data, test_labels = Space.get_slices(90000, 10000)

        with open("train_data", "wb") as trd, \
             open("train_labels", "wb") as trl, \
             open("test_data", "wb") as ted, \
             open("test_labels", "wb") as tel:
            pickle.dump(train_data, trd)
            pickle.dump(train_labels, trl)
            pickle.dump(test_data, ted)
            pickle.dump(test_labels, tel)

    classifier = KNeighborsClassifier(n_neighbors = 3)
    classifier.fit(np.asarray(train_data), np.asarray(train_labels))
    '''predictions = classifier.predict(np.asarray(test_data))
    classes = [ "primary-study",
                "systematic-review",
                "structured-summary-of-systematic-review",
                "overview",
                "structured-summary-of-primary-study"]

    if len(test_labels) != len(predictions):
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

    assert len(test_labels) == len(predictions)
    hits = 0
    for l, p in zip(test_labels, predictions):
        if l == p:
            hits += 1
    print(hits / len(test_labels))

    recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0,class_dimension)))
    recall_sum = 0
    for i in range(0,class_dimension):
        rcl = recall(i)
        recall_sum += rcl
        print('Recall {}: {:.5f}'.format(i, rcl))
    print()
    print('Recall mean: {:.5f}'.format(recall_sum/class_dimension))

    precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0,class_dimension)))
    precision_sum = 0
    for i in range(0,class_dimension):
        label_precision = precision(i)
        precision_sum += label_precision
        print('Precision {}: {:.5f}'.format(i, label_precision))
    print()
    print('Precision mean: {:.5f}'.format(precision_sum/class_dimension))'''

    for testy, label in zip(test_data, test_labels):
        print(label, classifier.predict_proba(np.asarray([testy])), classifier.predict(np.asarray([testy])))
