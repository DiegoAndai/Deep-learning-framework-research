## Evaluation protocol for january 10th reunion

### **Test 1:** Random, pvp and knn

To test what classificator does better, we'll use 5 embedding models namely Model 1, 3, 4, 5 and 6 (we had a problem with number 2). Specifications for these models are on [this folder](/Epistemonikos/LanguageModels/SkipGram/Advanced/TrainedModels). metrics to be used are accuracy, precision and recall (the former two both as the mean of every class and for each one)

Classifications will be done on two classes, systematic-review and primary-study. An evenly distributed dataset will be used for training and testing (50% of each class).

#### Classificators specifications:

- Random: using python's random an even decision will be made for every paper.

- KNN: [Scikit-learn implementation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) with k = 5 and default settings. The vector for each document is calculated bia a max pooling regarding the maximun value of each dimension of the document's first 10 words to appear on the model dictionary. Implementation [here](/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification).

- PVP: Add description

#### Hipothesis

The KNN model will have better results, as the PVP aproach calculates a centroid that's very likely to loose information on the high dimension level we are working.

#### Results

results will be graphed on:

- A bar chart for every metric, denoting the three measures (random, pvp and knn) for every model

