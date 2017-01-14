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

        with DocumentSpace.class_sess.as_default(), \
                    tf.device('/cpu:0'):

            self.language_model = tf.nn.l2_normalize(language_model,
                                                     1).eval()  # Word embeddings used to get vectors from text.

    def get_abs_vectors(self, papers):

        """ Generates a vector for every abstract to be classified.
        :parameter papers: JSON array (python list) containing
        the processed papers, represented as dicts."""


        labels = [paper["classification"] for paper in papers]  # ground truth

        print("Calculating abstracts' vectors...")
        parsed = 0
        n_abstracts = len(papers)
        print("started vector build")
        rel_freqs = []
        pooled_vectors = list()
        append_rel_freqs = rel_freqs.append
        lmo = self.lang_mod_order
        span = self.span
        asarr = np.asarray
        for paper in papers:  # 10000
            word_mtx = list()
            words = paper["abstract"].split()
            abs_count = collections.Counter(words)
            word_count = 0
            i = 0
            while word_count < self.span and i < len(words):
                word = words[i]
                if word in lmo:
                    index = self.lang_mod_order.index(word)
                else:
                    index = 0
                word_mtx.append(self.language_model[index])
                word_count += 1
                i += 1
            try:
                dim_mtx = np.transpose(word_mtx) #transposed : rows:value for every word at a given dimension
                pooled_vector = [max(dimension) for dimension in dim_mtx]
            except TypeError:
                pooled_vector = [0] * 10
            pooled_vectors.append(pooled_vector)
            if not parsed % 1000:
                print("{}/{}".format(parsed, n_abstracts))
            parsed += 1
        assert n_abstracts == parsed
        print("Finished calculating abstracts' vectors.")
        return list(zip(labels, pooled_vectors))


    @staticmethod
    def slice(vectors):
        data = [t[1] for t in vectors]
        labels = [t[0] for t in vectors]

        return data, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify papers.")
    parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
    parser.add_argument("--model_path", help="Path to a folder containing everything related to the model, namely "
                                             "embeddings and vocab files.",
                        required=True)
    parser.add_argument("--span", help="Number of words used for classification, counting from "
                                       "the start of the abstract", type=int, default=10)
    parser.add_argument("--KNN_papers_set", help="Path to the KNN paper set.",
                        required=True)

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

        args = parser.parse_args()

    else:
        args = parser.parse_args()

        path_to_model = args.model_path
        path_to_papers = args.KNN_papers_set
        join_path = os.path.join

        with open(join_path(path_to_model, "embeddings"), 'rb') as embeddings, \
                open(join_path(path_to_model, "vocab.txt")) as vocab, \
                open(join_path(path_to_papers, "train_papers")) as train_set, \
                open(join_path(path_to_papers, "test_papers")) as test_set:

            model = pickle.load(embeddings)
            model_order = [line.split()[0].strip("b'") for line in vocab]

            train = json.load(train_set)
            test = json.load(test_set)

        fthousand = []
        ps = 0
        sr = 0
        i = 0
        while ps < 5000 or sr < 5000:
            if test[i]["classification"] == "primary-study" and ps < 5000:
                fthousand.append(test[i])
                ps += 1
            elif test[i]["classification"] == "systematic-review" and sr < 5000:
                fthousand.append(test[i])
                sr += 1
            i += 1
        print(len(fthousand))

        Space = DocumentSpace(model, model_order, args.span)
        Space.train_vectors = Space.get_abs_vectors(train)
        Space.test_vectors = Space.get_abs_vectors(fthousand)
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


    classifier = KNeighborsClassifier(n_neighbors=args.K)
    print("fitting")
    classifier.fit(np.asarray(train_data), np.asarray(train_labels))
    print("predicting")
    predictions = classifier.predict(np.asarray(test_data))
    classes = [ "primary-study",
                 "systematic-review"]

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
