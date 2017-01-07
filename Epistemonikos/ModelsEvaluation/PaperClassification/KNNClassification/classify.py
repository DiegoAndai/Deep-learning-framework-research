from sklearn.neighbors import KNeighborsClassifier

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
        self.train_vectors = list()  # structure: [(type, vector)]
        self.test_vectors = list()  # structure: [(type, vector)]
        self.span = span
        self.lang_mod_order = lang_mod_order  # list of words of the language model (to know which
        # word an embedding represents)

        with DocumentSpace.class_sess.as_default():
            self.language_model = tf.nn.l2_normalize(language_model,
                                                     1).eval()  # Word embeddings used to get vectors from text.

    def get_abs_vectors(self, papers):

        """ Generates a vector for every abstract to be classified.
        :parameter papers: JSON array (python list) containing
        the processed papers, represented as dicts."""

        labels = [paper["classification"] for paper in papers]  # ground truth

        print("Calculating abstracts' vectors...")
        abs_vectors = list()
        parsed = 0
        n_abstracts = len(papers)
        print("started vector build")
        for paper in papers:
            abstract = paper["abstract"]
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
        print("Finished calculating abstracts' vectors.")
        return list(zip(labels, abs_vectors))

    @staticmethod
    def slice(vectors):
        data = [t[1] for t in vectors]
        labels = [t[0] for t in vectors]

        return data, labels


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
        parser.add_argument("--model_path", help="Path to a folder containing everything related to the model, namely "
                                                 "embeddings and vocab files.",
                            required=True)
        parser.add_argument("--span", help="Number of words used for classification, counting from "
                                           "the start of the abstract", type=int, default=10)
        parser.add_argument("--KNN_papers_set", help="Path to the KNN paper set.",
                            required=True)
        parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
        args = parser.parse_args()

        path_to_model = args.model_path
        path_to_papers = args.KNN_papers_set
        join_path = os.path.join

        with open(join_path(path_to_model, "embeddings")) as embeddings, \
                open(join_path(path_to_model, "vocab.txt")) as vocab, \
                open(join_path(path_to_papers, "train")) as train_set, \
                open(join_path(path_to_papers, "test")) as test_set:

            model = pickle.load(embeddings)
            model_order = [line.split()[0].strip("b'") for line in vocab]

            train = json.load(train_set)
            test = json.load(test_set)

        Space = DocumentSpace(model, model_order, args.span)
        Space.train_vectors = Space.get_abs_vectors(train_set)
        Space.test_vectors = Space.get_abs_vectors(test_set)
        train_data, train_labels = Space.slice(Space.train_vectors)
        test_data, test_labels = Space.slice(Space.test_vectors)

        with open("train_data", "wb") as trd, \
                open("train_labels", "wb") as trl, \
                open("test_data", "wb") as ted, \
                open("test_labels", "wb") as tel:
            pickle.dump(train_data, trd)
            pickle.dump(train_labels, trl)
            pickle.dump(test_data, ted)
            pickle.dump(test_labels, tel)

    classifier = KNeighborsClassifier(n_neighbors=args.K)  # FIXME: args may not be defined
    classifier.fit(np.asarray(train_data), np.asarray(train_labels))
    # '''predictions = classifier.predict(np.asarray(test_data))
    # classes = [ "primary-study",
    #             "systematic-review",
    #             "structured-summary-of-systematic-review",
    #             "overview",
    #             "structured-summary-of-primary-study"]
    #
    # if len(test_labels) != len(predictions):
    #     print("dimensions error. labels: {}, predictions: {}".format(len(test_labels),
    #                                                                  len(predictions)))
    #
    # class_dimension = len(classes)
    # conf_mtx = np.zeros([class_dimension, class_dimension])
    # for i in range(0, len(predictions)):
    #     predicted_class = classes.index(predictions[i])
    #     actual_class = classes.index(test_labels[i])
    #     conf_mtx[actual_class][predicted_class] += 1
    # np.set_printoptions(suppress=True)
    # print(conf_mtx)
    #
    # assert len(test_labels) == len(predictions)
    # hits = 0
    # for l, p in zip(test_labels, predictions):
    #     if l == p:
    #         hits += 1
    # print(hits / len(test_labels))
    #
    # recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0,class_dimension)))
    # recall_sum = 0
    # for i in range(0,class_dimension):
    #     rcl = recall(i)
    #     recall_sum += rcl
    #     print('Recall {}: {:.5f}'.format(i, rcl))
    # print()
    # print('Recall mean: {:.5f}'.format(recall_sum/class_dimension))
    #
    # precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0,class_dimension)))
    # precision_sum = 0
    # for i in range(0,class_dimension):
    #     label_precision = precision(i)
    #     precision_sum += label_precision
    #     print('Precision {}: {:.5f}'.format(i, label_precision))
    # print()
    # print('Precision mean: {:.5f}'.format(precision_sum/class_dimension))'''

    for testy, label in zip(test_data, test_labels):
        print(label, classifier.predict_proba(np.asarray([testy])), classifier.predict(np.asarray([testy])))
