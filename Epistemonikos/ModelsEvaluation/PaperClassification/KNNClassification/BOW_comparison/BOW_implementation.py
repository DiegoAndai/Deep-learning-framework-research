import argparse
import json
import os

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score




parser = argparse.ArgumentParser(description="Classify papers.")
parser.add_argument("--K", type=int, required=True, help="Number of nearest neighbours to consider")
parser.add_argument("--span", help="Number of words used for classification, counting from "
                                   "the start of the abstract", type=int, default=10)
parser.add_argument("--papers_set", help="Path to the paper set.",
                    required=True)
parser.add_argument("--vocab", help="Path to the vocabulary.",
                    required=True)
parser.add_argument("--distance_metric", default="minkowski", help="Metric to use to select nearest neighbours. "
                                                                   "Currently Minkowsky and dot product are "
                                                                   "implemented.")

args = parser.parse_args()
span = args.span
path_to_papers = args.papers_set
path_to_vocab = args.vocab
join_path = os.path.join

##LOAD DATA

vocabulary = list()

with open(join_path(path_to_papers, "train_papers")) as train_set, \
        open(join_path(path_to_papers, "test_papers")) as test_set, \
        open(join_path(path_to_vocab, "vocab.txt")) as vocab_file:

    train = json.load(train_set)
    test = json.load(test_set)

    for line in vocab_file:
        word = line.split()[0]
        vocabulary.append(word)

train_papers = [' '.join(t["abstract"].split()[:span]) for t in train]
train_labels = [t["classification"] for t in train]

test_papers = [' '.join(t["abstract"].split()[:span]) for t in test]
test_labels = [t["classification"] for t in test]

vocabulary = list(set(vocabulary))[:5000]

##VECTORIZE

index_split = len(train_papers)

vectorizer = CountVectorizer(vocabulary = vocabulary)

train_data = vectorizer.transform(train_papers)
test_data = vectorizer.transform(test_papers)




classifier = KNeighborsClassifier(n_neighbors=args.K, metric=args.distance_metric)
print("fitting")
classifier.fit(train_data, train_labels)
print("predicting")
predictions = list()
for paper in test_data:
    predictions.append(classifier.predict(paper))
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

output = ''
output += 'KNN classifier with k = {}\n'.format(args.K)
output += 'span = {}\n'.format(span)
output += 'Set: {}\n'.format(args.papers_set)
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


with open("output_BOW.txt", "w") as out_file:
    out_file.write(output)
