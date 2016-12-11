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

with open("../SkipGram/embedding", "rb") as embed_serialized:
    final_embeddings = pickle.load(embed_serialized)

with open("../SkipGram/reverse_dictionary", "rb") as reverse_dictionary_file:
    reverse_dictionary = pickle.load(reverse_dictionary_file)

labels = [reverse_dictionary[i].lower().strip() for i in range(len(reverse_dictionary))]

X = list()
Y = list()
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
        t1bool = t1 in labels
        t2bool = t2 in labels
        if t1bool and t2bool:
            t = RelatedTuple(t1, t2, pair[mean_index] / 1600)
            pairs.append(t)
            index = labels.index(t1)
            X.append(final_embeddings[index, :])
            index = labels.index(t2)
            Y.append(final_embeddings[index, :])

with open("wordsim_relatedness_goldstandard.txt", "r") as txt_file:
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
for i in range(len(pairs)):
    i_cosine = cosine[i][i]
    plt.scatter(pairs[i].relatedness, i_cosine, c = "k", s = 10)

    #print("pair terms: {}, {}. relatedness: {}. cosine: {}.".format(pairs[i].term1,
    #                                                               pairs[i].term2,
    #                                                               pairs[i].relatedness,
    #                                                               i_cosine))
    if pairs[i].term1 == "nausea":
        print(X[i])
        print(Y[i])
        print(np.dot(X[i], Y[i]))
        print(np.linalg.norm(X[i]) * np.linalg.norm(Y[i]))
        print(np.dot(X[i], Y[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(Y[i])))


    cosine_values.append(i_cosine)
    relatedness_values.append(pairs[i].relatedness)

print(pearsonr(cosine_values, relatedness_values))
print(spearmanr(cosine_values, relatedness_values))

'''print("low dim")
print(len(pairs))
max_ = 10000

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(final_embeddings[:max_, :])

X_low = list()
Y_low = list()
not_considered = list()
for pair in pairs:
    t1_index = labels.index(pair.term1)
    t2_index = labels.index(pair.term2)
    if t1_index < max_ and t2_index < max_:
        X_low.append(low_dim_embs[t1_index, :])
        Y_low.append(low_dim_embs[t2_index, :])
    else:
        not_considered.append(pairs.index(pair))


print(len(pairs))
cosine_values = []
relatedness_values = []
cosine = cosine_similarity(X_low, Y_low)
print(cosine.shape)
j = 0
for i in range(len(pairs)):
    if i not in not_considered:
        i_cosine = abs(cosine[j][j])
        plt.scatter(i_cosine, pairs[i].relatedness, c = "r", s = 7)

        print("pair terms: {}, {}. cosine: {}. relatedness: {}".format(pairs[i].term1,
                                                                       pairs[i].term2,
                                                                       i_cosine,
                                                                       pairs[i].relatedness))


        cosine_values.append(i_cosine)
        relatedness_values.append(pairs[i].relatedness)
        j += 1

print(pearsonr(cosine_values, relatedness_values))
print(spearmanr(cosine_values, relatedness_values))
'''
#plt.show()
