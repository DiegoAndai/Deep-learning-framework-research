from classify import DocumentSpace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pickle
import json
import collections
import os

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
    parser.add_argument("--distance_metric", default="minkowski", help="Metric to use to select nearest neighbours. "
                                                                       "Currently Minkowsky and dot product are "
                                                                       "implemented.")

    correct_classified = {"primary-study": [], "systematic-review": []}

    incorrect_classified = {"primary-study": [], "systematic-review": []}

    predictions = list()
    classes = ["primary-study", "systematic-review"]
    args = parser.parse_args()

    #using previous data
    if os.path.exists("predictions_proba") and os.path.exists("labels"):
        print("opening previous data for probability study")

        with open("predictions_proba", "rb") as pp_file, \
             open("labels", "rb") as lb_file:

            predictions_proba = pickle.load(pp_file)
            test_labels = pickle.load(lb_file)


        i = 0
        for prediction_proba, label in zip(predictions_proba, test_labels):

            prediction = ("primary-study" if prediction_proba[0] >= prediction_proba[1] \
                          else "systematic-review")
            index = (0 if prediction_proba[0] >= prediction_proba[1] \
                     else 1)
            predictions.append(prediction)

            if prediction != label:
                incorrect_classified[prediction].append(prediction_proba[index])
            else:
                correct_classified[prediction].append(prediction_proba[index])
            i += 1
            if i % 1000 == 0:
                print(i)

    else:
        if os.path.exists("test_data"):
            print("opening previous data for classification and then probability study")
            with open("train_data", "rb") as trd, \
                    open("train_labels", "rb") as trl, \
                    open("test_data", "rb") as ted, \
                    open("test_labels", "rb") as tel:
                train_data = pickle.load(trd)
                train_labels = pickle.load(trl)
                test_data = pickle.load(ted)
                test_labels = pickle.load(tel)


        else:
            print("starting from embeddings")


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

            Space = DocumentSpace(model, model_order, args.span)
            Space.train_vectors = Space.get_abs_vectors(train)
            Space.test_vectors = Space.get_abs_vectors(test)
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

        if args.distance_metric == "dot":
            args.distance_metric = np.dot

        classifier = KNeighborsClassifier(n_neighbors=args.K, metric=args.distance_metric, n_jobs=-1)
        print("fitting")
        classifier.fit(np.asarray(train_data), np.asarray(train_labels))
        print("predicting")

    ## PROBABILITIES:


        predictions_proba = list()

        i = 0
        for vector, label in zip(test_data, test_labels):

            prediction_proba = classifier.predict_proba(np.asarray(vector).reshape(1, -1))[0]
            prediction = ("primary-study" if prediction_proba[0] >= prediction_proba[1] \
                          else "systematic-review")
            index = (0 if prediction_proba[0] >= prediction_proba[1] \
                     else 1)
            predictions_proba.append(prediction_proba)
            predictions.append(prediction)

            if prediction != label:
                incorrect_classified[prediction].append(prediction_proba[index])
            else:
                correct_classified[prediction].append(prediction_proba[index])
            i += 1
            if i % 1000 == 0:
                print(i)

        #saving predictions and labels as this is slow to compute
        with open("predictions_proba", "wb") as pp_file, \
             open("labels", "wb") as lb_file:

            pickle.dump(predictions_proba, pp_file)
            pickle.dump(test_labels, lb_file)


    output = "OUTPUT LOG:\n"
    for _class in classes:
        print("analizing {}".format(_class))
        try:
            output += "\n{}:\n".format(_class.upper())
            output += "{} max probability when wrong: {:.5f}\n".format(_class, np.amax(incorrect_classified[_class]))
            output += "{} min probability when wrong: {:.5f}\n".format(_class, np.amin(incorrect_classified[_class]))
            output += "{} avg probability when wrong: {:.5f}\n".format(_class, np.mean(incorrect_classified[_class]))
            fig, ax = plt.subplots(1,1)
            histo_info = plt.hist(np.hstack(incorrect_classified[_class]), bins = 5)
            counts = [int(count) for count in histo_info[0]]
            bin_limits = [limit * args.K for limit in histo_info[1]]
            plt.xticks(histo_info[1], bin_limits, rotation = 30)
            output += "\nIntervals for wrong {} (format |interval, count|):\n".format(_class)
            for i in range(len(counts)):
                output += "| {:.3f} - {:.3f}, {:d} |".format(bin_limits[i], bin_limits[i+1], counts[i])
            for label in ax.get_xticklabels()[::2]:
                label.set_visible(False)
            output += "\n\n"
            plt.savefig("{}_histo_wrong.png".format(_class))
            plt.clf()
            output += "{} max probability when right: {:.5f}\n".format(_class, np.amax(correct_classified[_class]))
            output += "{} min probability when right: {:.5f}\n".format(_class, np.amin(correct_classified[_class]))
            output += "{} avg probability when right: {:.5f}\n".format(_class, np.mean(correct_classified[_class]))
            fig, ax = plt.subplots(1,1)
            histo_info = plt.hist(np.hstack(correct_classified[_class]), bins = 5)
            counts = [int(count) for count in histo_info[0]]
            bin_limits = [limit * args.K for limit in histo_info[1]]
            plt.xticks(histo_info[1], bin_limits, rotation = 30)
            output += "\nIntervals for right {} (format |interval, count|):\n".format(_class)
            for i in range(len(counts)):
                output += "| {:.3f} - {:.3f}, {:d} |".format(bin_limits[i], bin_limits[i+1], counts[i])
            for label in ax.get_xticklabels()[::2]:
                label.set_visible(False)
            output += "\n\n"
            plt.savefig("{}_histo_right.png".format(_class))
            plt.clf()
            print(output)
        except (ValueError, IndexError) as error:
            output += "Couldn't compute data for {}, Error log: {}".format(_class, error)

# TODO : remember to fix decimals on intervals so the output looks better
# TODO : fixed bins

## METRICS:

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
    print('F1 score weighted average: {:.5f}'.format(f1))

    output += '\n\nMETRICS:\n\n'
    output += 'Model: {}\n'.format(args.model_path)
    output += 'KNN classifier with k = {}\n'.format(args.K)
    output += 'span = {}\n'.format(args.span)
    output += 'Set: {}\n'.format(args.KNN_papers_set)
    output += 'Accuracy : {}\n'.format(accuracy)
    output += "RECALL\n"
    for rcl in recall_list:
        output += 'Recall {}: {:.5f}\n'.format(rcl[0], rcl[1])
    output += 'Recall mean: {:.5f}\n'.format(recall_mean)
    output += 'Recall weighted average: {:.5f}\n'.format(micro_recall)
    output += "PRECISION\n"
    for pcsn in precision_list:
        output += 'Precision {}: {:.5f}\n'.format(pcsn[0], pcsn[1])
    output += 'Precision mean: {:.5f}\n'.format(precision_mean)
    output += 'Precision weighted average: {:.5f}\n'.format(micro_precision)
    output += 'F1 score weighted average: {:.5f}\n'.format(f1)
    output += 'CONFUSSION MATRIX\n'
    output += str(conf_mtx)



    with open("output_prob.txt", "w") as out_file:
        out_file.write(output)
