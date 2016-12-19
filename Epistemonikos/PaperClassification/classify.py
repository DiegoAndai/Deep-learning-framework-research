import collections
import tensorflow as tf
import numpy as np
import pickle
import json
from Epistemonikos.SkipGram.open_documents import PaperReader


class PVPClassifier:  # Pondered vector paper classifier

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
        self.language_model = language_model  # Word embeddings used to get vectors from text.
        self.lang_mod_order = lang_mod_order  # list of words of the language model (to know which
        # word an embedding represents).
        self.predictions = None  # Classification
        self.labels = None

    def get_ref_vectors(self, new_n_save=True):

        """Generates one reference vector for each paper type.
        :parameter new_n_save: when True, the vectors are calculated and saved to a file,
        otherwise they are obtained from reference_vectors file."""

        if not new_n_save:
            with open("reference_vectors", "rb") as rv:
                self.reference_vectors = pickle.load(rv)
            return

        reader = PaperReader(self.reference_papers)

        print("Calculating reference vectors...")
        document_vectors = list()

        for type_ in self.classes:

            reader.remove_all_filters()
            reader.apply_filter(type_)
            reader.generate_words_list(self.span)
            type_words = reader.words
            type_count = collections.Counter(type_words)

            vector = []
            for word in self.lang_mod_order:
                if word in type_count:
                    vector.append(type_count[word])
                    del type_count[word]
                else:
                    vector.append(0)

            freq = np.array(vector, ndmin=2)
            nwords = sum(vector)
            rel_freq = freq / nwords
            document_vectors.append(rel_freq)

        document_embeds = np.asarray([np.dot(vector, self.language_model) for vector in document_vectors])
        reference_vectors = np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])

        with open("reference_vectors", "wb") as rv:
            pickle.dump(reference_vectors, rv)

        self.reference_vectors = reference_vectors
        print("Finished reference vectors calculation.")

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
            rel_freq = freq / nwords
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


def get_n_papers(n, i=0):

    with open("../SkipGram/documents_array.json", "r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    return loaded[i:i + n]


if __name__ == '__main__':

    document_types = ["systematic-review",
                      "structured-summary-of-systematic-review",
                      "primary-study",
                      "overview",
                      "structured-summary-of-primary-study"]

    with open("../SkipGram/embedding", "rb") as e:
        model = pickle.load(e)

    with open("../SkipGram/count", "rb") as c:
        l_m_order = [w[0] for w in pickle.load(c)]

    with open("../SkipGram/documents_array.json", encoding="utf-8") as da:
        ref_papers = json.load(da)

    classifier = PVPClassifier(model, l_m_order, document_types, ref_papers)
    classifier.get_ref_vectors(new_n_save=False)
    classifier.get_abs_vectors(get_n_papers(200), new_n_save=False)
    classifier.classify()
    print(classifier.get_accuracy())
