import pickle
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from collections import namedtuple


RelatedTuple = namedtuple("RelatedTuple", ["term1", "term2", "relatedness"])

print("remember you should first prepare the data and run the training at least once")
option = input("if you've done this: would you want to calculate wiki or episte? (enter w or e): ")

while option != "w" and option != "e":
    print("remember you should first prepare the data and run the training at least once")
    option = input("if you've done this: would you want to calculate wiki or episte? (enter w or e): ")

if option == "w":
    embed_name = "wikiembedding"
else:
    embed_name = "episteembedding"

with open(embed_name, "rb") as embed_serialized:
    final_embeddings = pickle.load(embed_serialized)

with open("reverse_dictionary", "rb") as reverse_dictionary_file:
    reverse_dictionary = pickle.load(reverse_dictionary_file)

labels = [reverse_dictionary[i].lower().strip() for i in range(len(reverse_dictionary))]

X = list()
Y = list()
pairs = []
with open("../Relatedness/UMNSRS_relatedness.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter = ",", quoting=csv.QUOTE_NONNUMERIC)
    head = next(reader)
    t1_index = head.index("Term1")
    t2_index = head.index("Term2")
    mean_index = head.index("Mean")
    for pair in reader:
        t1 = pair[t1_index].lower()
        t2 = pair[t2_index].lower()
        t1bool = t1 in labels
        t2bool = t2 in labels
        if t1bool and t2bool:
            t = RelatedTuple(t1, t2, pair[mean_index] / 1600)
            pairs.append(t)
            index = labels.index(t1)
            X.append(final_embeddings[index, :])
            index = labels.index(t2)
            Y.append(final_embeddings[index, :])

with open("../Relatedness/wordsim_relatedness_goldstandard.txt", "r") as txt_file:
    reader = (line.split() for line in txt_file)
    for pair in reader:
        t1 = pair[0].lower()
        t2 = pair[1].lower()
        t1bool = t1 in labels
        t2bool = t2 in labels
        if t1bool and t2bool:
            t = RelatedTuple(t1, t2, float(pair[2]) / 10)
            pairs.append(t)
            index = labels.index(t1)
            X.append(final_embeddings[index, :])
            index = labels.index(t2)
            Y.append(final_embeddings[index, :])




cosine_values = []
relatedness_values = []
cosine = cosine_similarity(X, Y)

verbose = False

for i in range(len(pairs)):
    i_cosine = cosine[i][i]
    plt.scatter(pairs[i].relatedness, i_cosine, c = "k", s = 10)

    if verbose:
        print("pair terms: {}, {}. relatedness: {}. cosine: {}.".format(pairs[i].term1,
                                                                       pairs[i].term2,
                                                                       pairs[i].relatedness,
                                                                       i_cosine))


    cosine_values.append(i_cosine)
    relatedness_values.append(pairs[i].relatedness)

print(pearsonr(cosine_values, relatedness_values))
print(spearmanr(cosine_values, relatedness_values))

plt.show()
