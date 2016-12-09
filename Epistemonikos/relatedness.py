import pickle
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple

RelatedTuple = namedtuple("RelatedTuple", ["term1", "term2", "relatedness"])

with open("embedding", "rb") as embed_serialized:
    final_embeddings = pickle.load(embed_serialized)

with open("reverse_dictionary", "rb") as reverse_dictionary_file:
    reverse_dictionary = pickle.load(reverse_dictionary_file)

labels = [reverse_dictionary[i] for i in range(len(reverse_dictionary))]

pairs = []
with open("UMNSRS_relatedness.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter = ",", quoting=csv.QUOTE_NONNUMERIC)
    head = next(reader)
    t1_index = head.index("Term1")
    t2_index = head.index("Term2")
    mean_index = head.index("Mean")
    for pair in reader:
        t1 = pair[t1_index].lower()
        t2 = pair[t2_index].lower()
        t1b = t1 in labels
        t2b = t2 in labels
        if t1b and t2b:
            t = RelatedTuple(t1, t2, pair[mean_index] / 1600)
            pairs.append(t)

with open("wordsim_relatedness_goldstandard.txt", "r") as txt_file:
    reader = (line.split() for line in txt_file)
    for pair in reader:
        t1 = pair[0].lower()
        t2 = pair[1].lower()
        t1b = t1 in labels
        t2b = t2 in labels
        if t1b and t2b:
            t = RelatedTuple(t1, t2, float(pair[2]) / 10)
            pairs.append(t)


X = list()
Y = list()
for pair in pairs:
    t1_index = labels.index(pair.term1)
    X.append(final_embeddings[t1_index])

    t2_index = labels.index(pair.term2)
    Y.append(final_embeddings[t2_index])


cosine_values = []
relatedness_values = []
for i in range(len(pairs)):
    cosine = cosine_similarity(X[i].reshape(1, -1), Y[i].reshape(1, -1))[0][0]
    plt.scatter(cosine, pairs[i].relatedness, c = "k", s = 7)

    cosine_values.append(cosine)
    relatedness_values.append(pairs[i].relatedness)

print(pearsonr(cosine_values, relatedness_values))
plt.show()
