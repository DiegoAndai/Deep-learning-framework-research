## Evaluation protocol for january 10th reunion

### **Test 1:** Random, pvp and knn

To test what classifier does better, we'll use 5 embedding models namely Model 4, 5 and 6 (we had a problem with number 2). Specifications for these models are on [this folder](/Epistemonikos/LanguageModels/SkipGram/Advanced/TrainedModels). metrics to be used are accuracy, precision and recall (the former two both as the mean of every class and for each one)

Classifications will be done on two classes, systematic-review and primary-study. An evenly distributed dataset will be used for training and testing (50% of each class).

#### Classifiers specifications:

- Random: using python's random an even decision will be made for every paper.

- KNN: [Scikit-learn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) with k = 5 and default settings. The vector for each document is calculated via a max pooling regarding the maximun value of each dimension of the document's first 10 words. Implementation [here](/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification).

- PVP: Add description. Implementation [here](/Epistemonikos/ModelsEvaluation/PaperClassification/PVPClassification).

#### Hypothesis

The KNN model will have better results, as the PVP aproach calculates a centroid that's very likely to loose information on the high dimension level we are working.

#### Results

results will be graphed on:

- A bar chart for every metric, denoting the three measures (random, pvp and knn) for every model

### Test2: Number of neighbors for KNN

We need to test the influence of the number of neighbors to consider for the knn classification. This will be tested on models 1, 3, 4, 5 and 6. Metrics will be same as above. Numbers to be tested will be 5, 10, 50 and 100.

#### Hypothesis

Some middle value of k will be the best for the classification, too many will just add noise, and few neighbors don't have enough information.

#### Results

results will be graphed on:

- A line chart for every metric, with x = k, y = metric and lines = models.
