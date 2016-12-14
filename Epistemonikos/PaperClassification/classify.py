import collections
import tensorflow as tf
import numpy as np
import pickle
import json
from Epistemonikos.SkipGram.open_documents import PaperReader


def get_vectors(document_types, limit_abstracts=None):

    """Generates one vector for each paper type
    and one vector for each abstract"""

    with open("../SkipGram/embedding", "rb") as embed_serialized:
        final_embeddings = pickle.load(embed_serialized)

    with open("../SkipGram/documents_array.json", "r") as json_file:
        loaded = json.load(json_file)

    with open("../SkipGram/count", "rb") as count_file:
        count = pickle.load(count_file)

    reader = PaperReader(loaded)

    print("calculating abs")
    abs_vectors = list()
    parsed = 0
    n_abstracts = len(reader)
    for abstract in reader:
        abs_count = collections.Counter(abstract[:limit_abstracts])
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
        abs_vector = np.dot(rel_freq, final_embeddings).squeeze()  # check shape
        abs_vectors.append(abs_vector)
        parsed += 1
        if not parsed % 10000:
            print("{}/{}".format(parsed, n_abstracts))
    abs_vectors = np.asarray(abs_vectors)

    print("calculating refs")
    document_vectors = list()
    for type_ in document_types:
        reader.remove_all_filters()
        reader.apply_filter(type_)
        reader.generate_words_list(limit_abstracts)
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
    return abs_vectors, np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])


def classify(reference_vecs, abs_to_classify, types):

    with tf.Graph().as_default() as classification_graph:
        ref_vecs = tf.placeholder(tf.float32, shape=[None, None])  # vectors to decide classification. Shape depends
        # on number of document types and embeddings dimensionality.

        to_classify = tf.placeholder(tf.float32, shape=[None, None])  # vectors representing abstracts. Shape depends
        # on number of papers to classify and embeddings dimensionality

        similarity = tf.matmul(to_classify, ref_vecs, transpose_b=True)  # shape=[papers, document types]

        # no argsort in TensorFlow!!!!!

        init = tf.initialize_all_variables()

    with tf.Session(graph=classification_graph) as session:

        init.run()
        sim = session.run(similarity, feed_dict={ref_vecs: reference_vecs, to_classify: abs_to_classify})

    return [types[(-row).argsort()[0]] for row in sim]  # list with the classification for abs_to_classify


if __name__ == '__main__':

    document_types = ["systematic-review",
                      "structured-summary-of-systematic-review",
                      "primary-study",
                      "overview",
                      "structured-summary-of-primary-study"]

    option = input("0 -> fresh\n"
                   "else -> load\n")
    if option == "0":
        vecs = abs_vecs, ref_vecs = get_vectors(document_types, limit_abstracts=10)

        with open("abs_refs_vecs", "wb") as arv:
            pickle.dump(vecs, arv)
    else:
        with open("abs_refs_vecs", "rb") as arv:
            abs_vecs, ref_vecs = pickle.load(arv)

    print(abs_vecs.shape, ref_vecs.shape)

    classify(ref_vecs, abs_vecs, document_types)
