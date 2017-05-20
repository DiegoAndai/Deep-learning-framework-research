from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from error_finder import MaxPoolLab, DocLab

import numpy as np
import tensorflow as tf
import argparse
import pickle
import json
import collections
import os
from random import randint, choice


class DocumentSpace:
    class_sess = tf.Session()

    def __init__(self, language_model, lang_mod_order, span):
        self.train_vectors = list()  # structure: [(type, vector)]
        self.test_vectors = list()  # structure: [(type, vector)]
        self.span = span
        self.lang_mod_order = lang_mod_order  # list of words of the language model (to know which
        # word an embedding represents)
        self.max_pool_lab = MaxPoolLab()
        self.doc_lab = DocLab()

        with DocumentSpace.class_sess.as_default(), \
                    tf.device('/cpu:0'):

            self.language_model = tf.nn.l2_normalize(language_model,
                                                     1).eval()  # Word embeddings used to get vectors from text.

        #To shuffle uncomment:
        #print('Shuffling (add flag to this later)')
        #np.random.shuffle(self.language_model)

        #For random matrix uncomment:
        #print('Random matrix')
        #self.language_model = np.random.rand(*np.shape(self.language_model))

    def shuffle(self, words):
        print('Shuffling')
        to_shuffle_from = []
        to_shuffle_to = []
        c = 0
        for word in words:
            if word in self.lang_mod_order:
                c += 1
                index = self.lang_mod_order.index(word)
                to = randint(0, len(self.language_model)-1)
                i = self.language_model[index]
                j = self.language_model[to]
                self.language_model[index] = j
                self.language_model[to]= i
        print("shufled {} words".format(c))



    def get_abs_vectors(self, papers, mp_analysis = False, use_stopwords = False,
                              restricted_dict = None, counting = False):

        """ Generates a vector for every abstract to be classified.
        :parameter papers: JSON array (python list) containing
        the processed papers, represented as dicts.

        mp_analysis is a flag to save info of the process for later study
        counting is a flag to save word count due to restriction"""


        labels = [paper["classification"] for paper in papers]  # ground truth


        print("Calculating abstracts' vectors...")
        parsed = 0
        n_abstracts = len(papers)
        print("started vector build")
        rel_freqs = []
        pooled_vectors = list()
        append_rel_freqs = rel_freqs.append
        hash_table = {word : index for (index, word) in enumerate(self.lang_mod_order)}
        lmo = self.lang_mod_order
        span = self.span
        asarr = np.asarray
        for paper in papers:

            word_mtx = list()
            words = paper["abstract"].split()
            if counting:
                _id = paper["id"]
                word_count_lab = 0
            #for error study#
            #if restricted_dict:
            #    words = [word if word in restricted_dict else "UNK" for word in words]
            #################
            abs_count = collections.Counter(words)
            word_count = 0
            i = 0
            while word_count < self.span and i < len(words):
                word = words[i]
                ## for error study:
                if restricted_dict:
                    lookup = restricted_dict
                else:
                    lookup = hash_table
                ##
                if word in lookup and word in hash_table:
                    index = hash_table[word]
                    word_count_lab += 1
                else:
                    index = 0
                word_mtx.append(self.language_model[index])
                word_count += 1
                i += 1
            try:
                pooled_vector = np.amax(np.asarray(word_mtx), axis = 0)
            except ValueError: #no words
                pooled_vector = self.language_model[0] #unk

            #except TypeError:
            #    pooled_vector = np.zeros(shape = 500)
            pooled_vectors.append(pooled_vector)
            if counting:
                self.doc_lab.add_doc(_id, word_count_lab)
            if not parsed % 10000:
                print("--->{}/{}".format(parsed, n_abstracts), end = "\r")
            parsed += 1
        print("--->{}/{}".format(parsed, n_abstracts))
        assert n_abstracts == parsed
        print("Finished calculating abstracts' vectors.")
        return list(zip(labels, pooled_vectors))


    @staticmethod
    def slice(vectors):
        data = [t[1] for t in vectors]
        labels = [t[0] for t in vectors]

        return np.asarray(data), labels


#if __name__ == "__main__":
def main(restrict_k, save_id, restrict_random = False):
    parser = argparse.ArgumentParser(description="Classify papers.")
    parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
    parser.add_argument("--model_path", help="Path to a folder containing everything related to the model, namely "
                                             "embeddings and vocab files.",
                        required=True)
    parser.add_argument("--span", help="Number of words used for classification, counting from "
                                       "the start of the abstract", type=int, default=10)
    parser.add_argument("--KNN_papers_set", help="Path to the KNN paper set.",
                        required=True)
    parser.add_argument("--distance_metric", default="minkowski", help="Metric to use to select nearest neighbours. "
                                                                       "Currently Minkowsky and dot product are "
                                                                       "implemented.")

    args = parser.parse_args()

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

        #for error study#
        with open("Word_proba/proba_ratio_results_denis_method.json", "r") as restricted_json:
            restricted_dict = json.load(restricted_json)
        if restrict_k:
            if restrict_random:
                choosen_list = []
                for _ in range(restrict_k * 2):
                    choosen = choice(restricted_dict)
                    while choosen in choosen_list:
                        choosen = choice(restricted_dict)
                    choosen_list.append(choosen)
                restricted_dict = choosen_list
            else:
                restricted_dict = restricted_dict[:restrict_k] + restricted_dict[-restrict_k:]
        restricted_dict = [word_info[0] for word_info in restricted_dict]
        print(len(restricted_dict))

        #################

        Space = DocumentSpace(model, model_order, args.span)
        #for shuffling only
        #Space.shuffle(restricted_dict)
        ####################
        Space.train_vectors = Space.get_abs_vectors(train, restricted_dict = restricted_dict, counting = True)
        Space.doc_lab.save("training_{}k_{}id".format(restrict_k, save_id))
        Space.doc_lab.reset()
        Space.test_vectors = Space.get_abs_vectors(test, restricted_dict = restricted_dict, counting = True)
        Space.doc_lab.save("testing_{}k_{}id".format(restrict_k, save_id))
        train_data, train_labels = Space.slice(Space.train_vectors)
        test_data, test_labels = Space.slice(Space.test_vectors)

        '''with open("train_data", "wb") as trd, \
                open("train_labels", "wb") as trl, \
                open("test_data", "wb") as ted, \
                open("test_labels", "wb") as tel:
            pickle.dump(train_data, trd)
            pickle.dump(train_labels, trl)
            pickle.dump(test_data, ted)
            pickle.dump(test_labels, tel)'''

            #NOTE: use the dump above only with a capable computer to save time consuming data that can be rehused

    if args.distance_metric == "dot":
        args.distance_metric = np.dot

    classifier = KNeighborsClassifier(n_neighbors=args.K, metric=args.distance_metric, n_jobs=-1)
    print("fitting")
    classifier.fit(np.asarray(train_data), np.asarray(train_labels))
    print("predicting")
    predictions = list()
    i = 0
    dict_out = {"correct" : {"primary-study": [], "systematic-review": []},
                "incorrect": {"primary-study": [], "systematic-review": []}}
    for paper_vector, paper in zip(test_data, test):
        prediction = classifier.predict(paper_vector.reshape(1, -1))

        if prediction == paper["classification"]:
            dict_out["correct"][paper["classification"]].append(paper_vector.tolist())
        else:
            dict_out["incorrect"][paper["classification"]].append(paper_vector.tolist())

        predictions.append(prediction)
        i += 1
        if not i % 1000:
	           print('--->{}/{}'.format(i, len(test_labels)), end = "\r")
    print('--->{}/{}'.format(i, len(test_labels)))

    with open("dict_out", "w") as json_out:
        json.dump(dict_out, json_out)

    classes = ["primary-study", "systematic-review"]

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
    print('')
    accuracy = (sum(conf_mtx[i][i] for i in range(0, len(classes)))/len(predictions))
    print('Accuracy: {}'.format(accuracy))

    #Uncomment below for more specific metrics
    '''recall = lambda i: (conf_mtx[i][i]/sum(conf_mtx[i][j] for j in range(0,class_dimension)))
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
    print('Recall macro average: {:.5f}'.format(recall_mean))
    micro_recall = recall_score(test_labels, predictions, average='weighted')
    print('Recall weighted average: {:.5f}'.format(micro_recall))

    precision = lambda i: (conf_mtx[i][i]/sum(conf_mtx[j][i] for j in range(0, class_dimension)))
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
    print('Precision macro average: {:.5f}'.format(precision_mean))
    micro_precision = precision_score(test_labels, predictions, average='weighted')
    print('Precision weighted average: {:.5f}'.format(micro_precision))

    f1 = f1_score(test_labels, predictions, average='weighted')
    print('F1 score weighted average: {:.5f}'.format(f1))'''

    output = ''
    output += 'Model: {}\n'.format(args.model_path)
    output += 'KNN classifier with k = {}\n'.format(args.K)
    output += 'span = {}\n'.format(args.span)
    output += 'Set: {}\n'.format(args.KNN_papers_set)
    output += 'Accuracy : {}\n'.format(accuracy)
    output += "RECALL\n"
    '''for rcl in recall_list:
        output += 'Recall {}: {:.5f}\n'.format(rcl[0], rcl[1])
    output += 'Recall mean: {:.5f}\n'.format(recall_mean)
    output += 'Recall weighted average: {:.5f}\n'.format(micro_recall)
    output += "PRECISION\n"
    for pcsn in precision_list:
        output += 'Precision {}: {:.5f}\n'.format(pcsn[0], pcsn[1])
    output += 'Precision mean: {:.5f}\n'.format(precision_mean)
    output += 'Precision weighted average: {:.5f}\n'.format(micro_precision)
    output += 'F1 score weighted average: {:.5f}\n'.format(f1)'''
    output += 'CONFUSSION MATRIX\n'
    output += str(conf_mtx)
    output += Space.max_pool_lab.infographic_from_results()

    #Save results for later analysis:
    with open("test_output_no_exclusives_{}.txt".format(restrict_k), "w") as out_file: #add something to distinguish
        out_file.write(output)

    return accuracy, conf_mtx, restricted_dict

    '''with open("Max_pool_lab_results{}".format(save_id), "wb") as out_mpl:
        pickle.dump(Space.max_pool_lab.obtain_results(), out_mpl)'''

    '''with open("predictions_proba{}".format(save_id), "wb") as out_pp:
        pickle.dump(predictions, out_pp)

    with open("labels{}".format(save_id), "wb") as out_lbl:
        pickle.dump(test_labels, out_lbl)'''
