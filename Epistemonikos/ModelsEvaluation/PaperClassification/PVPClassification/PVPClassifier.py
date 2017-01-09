import collections
import tensorflow as tf
import numpy as np
import pickle
import json
import sys
sys.path.insert(0, 'PaperProcessing')
from open_documents import PaperReader
from tabulate import tabulate


class PVPClassifier:  # Pondered vector paper classifier

    class_sess = tf.Session()

    # Definition of ONE TensorFlow classification graph to use between instances. One graph to rule them all.
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        ref_vecs = tf.placeholder(tf.float32, shape=[None, None])  # vectors to decide classification. Shape depends
        # on number of document types and embeddings dimensionality.

        to_classify = tf.placeholder(tf.float32, shape=[None, None])  # vectors representing abstracts. Shape depends
        # on number of papers to classify and embeddings dimensionality

        similarity = tf.matmul(to_classify, ref_vecs, transpose_b=True)  # shape=[papers, document types]

        init_op = tf.global_variables_initializer()

    def __init__(self, language_model, lang_mod_order, classes, reference_papers, span=10, span_start=0):
        self.span = span  # How many words to consider from the abstracts to classify them.
        self.span_start = span_start  # The word index in the abstract from which the span starts. NOT USED.
        self.reference_vectors = None  # Vectors used as classification reference for abstracts' vectors.
        self.abstracts_vectors = None  # Vectors representing the abstracts (or parts of them)
        # of the papers to be classified.
        self.reference_papers = reference_papers  # List with papers (dictionaries) used to generate reference vectors.
        self.classes = classes

        self.lang_mod_order = lang_mod_order  # list of words of the language model (to know which
        # word an embedding represents).
        self.predictions = None  # Classification
        self.labels = None
        with PVPClassifier.class_sess.as_default():
            self.language_model = tf.nn.l2_normalize(language_model,
                                                     1).eval()  # Word embeddings used to get vectors from text.

    def get_ref_vectors(self, new_n_save=True, how=0):

        """Generates one reference vector for each paper type.
        :parameter new_n_save: when True, the vectors are calculated and saved to a file,
        otherwise they are obtained from reference_vectors file.
        :parameter how: non negative integer indicating the way the vectors should be calculated.
         0: ponder the embeddings by their relative frequencies in the span of the abstract.
         1: maxpooling from each column of a matrix composed by the vectors of the words in the abstracts' span.
        """

        if how == 0:
            if not new_n_save:
                with open("reference_vectors", "rb") as rv:
                    self.reference_vectors = pickle.load(rv)
                return


            print("Calculating reference vectors...")
            document_vectors = list()

            for type_ in self.classes:

                reader = PaperReader(self.reference_papers, abstracts_min=self.span, filters=[type_])
                reader.generate_words_list(limit_abstracts=self.span)
                type_words = reader.words
                type_count = collections.Counter(type_words)

                vector = []  # list with the frequencies of words for paper type type_
                for word in self.lang_mod_order:
                    if word in type_count:
                        vector.append(type_count[word])
                        del type_count[word]
                    else:
                        vector.append(0)

                freq = np.array(vector, ndmin=2)
                nwords = sum(vector)
                rel_freq = freq / nwords if nwords else freq
                document_vectors.append(rel_freq)

            document_embeds = np.asarray([np.dot(vector, self.language_model) for vector in document_vectors])
            reference_vectors = np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])

            with open("reference_vectors", "wb") as rv:
                pickle.dump(reference_vectors, rv)

            self.reference_vectors = reference_vectors
            print("Finished reference vectors calculation.")

        elif how == 1:


            # max_pool_ref_graph = tf.Graph()
            # with max_pool_ref_graph.as_default():
            #     tf.nn.embedding_lookup()

            with tf.Session().as_default(), tf.device('/cpu:0'):
                for cls in self.classes:
                    for paper in self.reference_papers:
                        if paper["classification"] == cls:
                            paper_words = paper["abstract"].split(' ')[:self.span]

                            word_indices = np.zeros(self.span, dtype=np.int32)
                            i = 0
                            while i < self.span:
                                try:
                                    word_indices[i] = self.lang_mod_order.index(paper_words[i])
                                except ValueError:
                                    word_indices[i] = 0
                                i += 1

                            paper_vectors = tf.nn.embedding_lookup(self.language_model, word_indices).eval()


    def get_abs_vectors(self, to_classify, new_n_save=True):

        """ Generates a vector for every abstract to be classified.
        :parameter to_classify: JSON array (python list) containing
        the abstracts (dicts) of the papers to be classified.
        :parameter new_n_save: when True, save the newly calculated vectors
        to a file, when False, get the vectors from a file."""

        self.labels = [paper["classification"] for paper in to_classify]  # ground truth

        if not new_n_save:
            with open("abstracts_vectors", "rb") as av:
                self.abstracts_vectors = pickle.load(av)
            return

        print("Calculating abstracts' vectors...")
        reader = PaperReader(to_classify)
        abs_vectors = list()
        parsed = 0
        n_abstracts = len(reader)
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
            if not parsed % 10000:
                print("{}/{}".format(parsed, n_abstracts))
        assert n_abstracts == parsed
        assert len(abs_vectors) == n_abstracts
        abs_vectors = np.asarray(abs_vectors)

        with open("abstracts_vectors", "wb") as av:
            pickle.dump(abs_vectors, av)

        self.abstracts_vectors = abs_vectors
        print("Finished calculating abstracts' vectors.")

    def classify(self):

        print("Classifying papers...")
        with tf.Session(graph=PVPClassifier.classification_graph) as session:
            PVPClassifier.init_op.run()
            sim = session.run(PVPClassifier.similarity,
                              feed_dict={PVPClassifier.ref_vecs: self.reference_vectors,
                                         PVPClassifier.to_classify: self.abstracts_vectors})

        # list with the classification for abs_to_classify (no argsort in TensorFlow!!!!!)
        self.predictions = [self.classes[(-row).argsort()[0]] for row in sim]
        print("Papers classified.")

    def get_accuracy(self):

        assert len(self.labels) == len(self.predictions)
        hits = 0
        for l, p in zip(self.labels, self.predictions):
            if l == p:
                hits += 1
        return hits / len(self.labels)

    def get_conf_matrix(self):
        if len(self.labels) != len(self.predictions):
            print("dimensions error. labels: {}, predictions: {}".format(len(self.labels),
                                                                         len(self.predictions)))

        class_dimension = len(self.classes)
        conf_mtx = np.zeros([class_dimension, class_dimension])
        for i in range(0, len(self.predictions)):
            predicted_class = self.classes.index(self.predictions[i])
            actual_class = self.classes.index(self.labels[i])
            conf_mtx[actual_class][predicted_class] += 1
        np.set_printoptions(suppress=True)
        return conf_mtx

    def get_conf_mat_pretty(self):
        matrix = self.get_conf_matrix().tolist()
        for cls in range(len(self.classes)):
            matrix[cls].insert(0, self.classes[cls])

        return tabulate(matrix, headers=[''] + self.classes)

    def recalls(self, verbose = True):
        conf_mtx = self.get_conf_matrix()
        class_dimension = len(self.classes)
        recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0, class_dimension)))
        for i in range(0, class_dimension):
            rcl = recall(i)
            # if verbose:
            #     print('Recall for {}: {:.5f}'.format(self.classes[i], rcl))
            yield self.classes[i], rcl

    def get_avg_recall(self):
        avg_rcl = 0
        for _, rcl in self.recalls():
            avg_rcl += rcl
        return avg_rcl/len(self.classes)

    def precisions(self, verbose = True):
        conf_mtx = self.get_conf_matrix()
        class_dimension = len(self.classes)
        precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0, class_dimension)))
        precision_sum = 0
        for i in range(0, class_dimension):
            label_precision = precision(i)
            if not np.isnan(label_precision):
                precision_sum += label_precision
            # if verbose:
            #     print('Precision for {}: {:.5f}'.format(self.classes[i], label_precision))
            yield self.classes[i], label_precision

    def get_avg_precision(self):
        avg_prec = 0
        for cls, prec in self.precisions():
            avg_prec += prec
        return avg_prec/len(self.classes)


def get_n_papers(n, path, i=0):

    with open(path, "r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    return loaded[i:i + n]
