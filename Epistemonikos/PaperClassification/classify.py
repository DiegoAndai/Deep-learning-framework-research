import collections
import tensorflow as tf
import numpy as np
import pickle
import json
from Epistemonikos.SkipGram.open_documents import PaperReader


def get_reference_vectors(limit_abstracts=False):

    """Generates one vector for each paper type"""

    with open("../SkipGram/embedding", "rb") as embed_serialized:
        final_embeddings = pickle.load(embed_serialized)

    with open("../SkipGram/documents_array.json", "r") as json_file:
        loaded = json.load(json_file)

    with open("../SkipGram/count", "rb") as count_file:
        count = pickle.load(count_file)

    document_types = ["systematic-review",
                      "structured-summary-of-systematic-review",
                      "primary-study",
                      "overview",
                      "structured-summary-of-primary-study"]

    document_vectors = list()
    reader = PaperReader(loaded)

    for type_ in document_types:
        reader.remove_all_filters()
        reader.apply_filter(type_)
        reader.generate_words_list(limit_abstracts) if limit_abstracts else reader.generate_words_list()
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
    return np.asarray([matrix for wrapped_matrix in document_embeds for matrix in wrapped_matrix])


def classify(ref_vecs):
    pass


if __name__ == '__main__':

    with open("ref_vec", "wb") as rv:
        ref_vecs = get_reference_vectors(10)
        pickle.dump(ref_vecs)
