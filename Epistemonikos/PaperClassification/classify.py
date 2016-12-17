import collections
import tensorflow as tf
import numpy as np
import pickle
import json
import sys
from Epistemonikos.SkipGram.open_documents import PaperReader


def get_ref_vectors(document_types, abstracts_limit=None, save=True):

    """Generates one reference vector for each paper type."""

    with open("../SkipGram/embedding", "rb") as embed_serialized:
        final_embeddings = pickle.load(embed_serialized)

    with open("../SkipGram/documents_array.json", "r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    with open("../SkipGram/count", "rb") as count_file:
        count = pickle.load(count_file)

    reader = PaperReader(loaded)

    print("calculating refs")
    document_vectors = list()

    for type_ in document_types:

        reader.remove_all_filters()
        reader.apply_filter(type_)
        reader.generate_words_list(abstracts_limit)
        type_words = reader.words
        type_count = collections.Counter(type_words)

        vector = []
        for word, _ in count:
            if word in type_count:
                vector.append(type_count[word])
                del type_count[word]
            else:
                vector.append(0)

        freq = np.array(vector, ndmin=2)
        nwords = sum(vector)
        rel_freq = freq / nwords
        document_vectors.append(rel_freq)

    document_embeds = np.asarray([np.dot(vector, final_embeddings) for vector in document_vectors])
    reference_vectors = np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])

    if save:
        with open("reference_vectors", "wb") as rv:
            pickle.dump(reference_vectors, rv)

    return reference_vectors


def get_abs_vectors(to_classify, abstracts_limit=None, save=True):

    """ Generates a vector for every abstract to be classified.
    :parameter to_classify: JSON array (python list) containing
    the abstracts (dicts) of the papers to be classified.
    :parameter abstracts_limit: number of words to consider from
    the beginning of the abstracts.
    :parameter save: save the vectors to a file."""

    with open("../SkipGram/embedding", "rb") as embed_serialized:
        final_embeddings = pickle.load(embed_serialized)

    with open("../SkipGram/count", "rb") as count_file:
        count = pickle.load(count_file)

    reader = PaperReader(to_classify)
    print("calculating abs")
    abs_vectors = list()
    parsed = 0
    n_abstracts = len(reader)
    for abstract in reader:
        abs_count = collections.Counter(abstract[:abstracts_limit])
        vector = []
        for word, _ in count:
            if word in abs_count:
                vector.append(abs_count[word])
                del abs_count[word]
            else:
                vector.append(0)
        freq = np.array(vector, ndmin=2)
        nwords = sum(vector)
        rel_freq = freq / nwords
        abs_vector = np.dot(rel_freq, final_embeddings).squeeze()
        abs_vectors.append(abs_vector)
        parsed += 1
        if not parsed % 10000:
            print("{}/{}".format(parsed, n_abstracts))
    assert n_abstracts == parsed
    assert len(abs_vectors) == n_abstracts
    abs_vectors = np.asarray(abs_vectors)

    if save:
        with open("abstracts_vectors", "wb") as av:
            pickle.dump(abs_vectors, av)

    return abs_vectors


def classify(reference_vecs, abs_to_classify, types):

    with tf.Graph().as_default() as classification_graph:
        ref_vecs = tf.placeholder(tf.float32, shape=[None, None])  # vectors to decide classification. Shape depends
        # on number of document types and embeddings dimensionality.

        to_classify = tf.placeholder(tf.float32, shape=[None, None])  # vectors representing abstracts. Shape depends
        # on number of papers to classify and embeddings dimensionality

        similarity = tf.matmul(to_classify, ref_vecs, transpose_b=True)  # shape=[papers, document types]

        init = tf.global_variables_initializer()

    with tf.Session(graph=classification_graph) as session:

        init.run()
        sim = session.run(similarity, feed_dict={ref_vecs: reference_vecs, to_classify: abs_to_classify})

    return [types[(-row).argsort()[0]] for row in sim]  # list with the classification for abs_to_classify
    # no argsort in TensorFlow!!!!!


def get_n_papers(n, i=0):

    with open("../SkipGram/documents_array.json", "r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)

    return loaded[i:i + n]


def get_accuracy(labels, predictions):

    assert len(labels) == len(predictions)
    hits = 0
    for l, p in zip(labels, predictions):
        if l == p:
            hits += 1
    return hits/len(labels)


if __name__ == '__main__':

    document_types = ["systematic-review",
                      "structured-summary-of-systematic-review",
                      "primary-study",
                      "overview",
                      "structured-summary-of-primary-study"]

    words = 10

    option = input("0 -> all fresh\n"
                   "1 -> fresh abs, load refs\n"
                   "2 -> fresh refs, load abs\n"
                   "3 -> load all\n")

    papers_to_classify = get_n_papers(1000, 10000)

    if option == "0":
        ref_vecs = get_ref_vectors(document_types, abstracts_limit=words)
        abs_vecs = get_abs_vectors(papers_to_classify, abstracts_limit=words)

    elif option == "1":
        with open("reference_vectors", "rb") as rv:
            ref_vecs = pickle.load(rv)
        abs_vecs = get_abs_vectors(papers_to_classify, abstracts_limit=words)

    elif option == "2":
        with open("abstracts_vectors", "rb") as av:
            abs_vecs = pickle.load(av)
        ref_vecs = get_ref_vectors(document_types, abstracts_limit=words)

    elif option == "3":
        with open("reference_vectors", "rb") as rv, open("abstracts_vectors", "rb") as av:
            ref_vecs = pickle.load(rv)
            abs_vecs = pickle.load(av)
    else:
        print("unkown option")
        sys.exit()

    print(ref_vecs.shape, abs_vecs.shape)

    labels = [paper["classification"] for paper in papers_to_classify]  # ground truth; actual classification of papers
    predictions = classify(ref_vecs, abs_vecs, document_types)  # predicted classification
    print(len(labels), len(predictions))
    print(get_accuracy(labels, predictions))
