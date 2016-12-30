# TO-DO:
# - define the graph only once and not each time predict_word is called.
# - define predicted_word_ids inside the graph
import pickle
import numpy as np
import tensorflow as tf

PATH_TO_EMBEDDINGS = "../SkipGram/"


def predict_word(wa, wb, wc):

    with open(PATH_TO_EMBEDDINGS + "embedding", 'rb') as embedding, \
            open(PATH_TO_EMBEDDINGS + "reverse_dictionary", 'rb') as rd:
        embeddings = tf.constant(pickle.load(embedding))
        reverse_dictionary = pickle.load(rd)

    dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))

    word_ids = tf.constant(np.array([dictionary[wa], dictionary[wb], dictionary[wc]]))
    wabc_vectors = tf.nn.embedding_lookup(embeddings, word_ids)
    wa_vector, wb_vector, wc_vector = wabc_vectors[0, :], wabc_vectors[1, :], wabc_vectors[2, :]
    result_vector = wc_vector + (wb_vector - wa_vector)
    similarity = tf.matmul(tf.expand_dims(result_vector, 0), embeddings, transpose_b=True)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        sim = session.run(similarity)
        predicted_word_ids = (-sim[0, :]).argsort()[:10]
        predicted_words = [reverse_dictionary[predicted_word_id] for predicted_word_id in predicted_word_ids]
        print("{} is to {} as {} is to {}".format(wa, wb, wc, predicted_words))
        return predicted_words


if __name__ == '__main__':

    predict_word("health", "sickness", "good")  # "evil" or similar is expected
    predict_word("paris", "france", "rome")  # italy
    predict_word("big", "bigger", "small")
    predict_word("copper", "cu", "gold")

