## Probability Study for KNN Classification

With this, we want to analyze how the classifier's confidence behaves in relation to it's right/wrong rate. Here, confidence is measured in number of neighbors of the predicted class. Our hipothesis is, as the ratio of neighbors becomes greater, the classifier should get more predictions right. 

Ratio here is calculated as (predicted class neighbors/total neighbors), so we expect to find a correlation between this measurement and correct predictions.

We tested model 4 on a uneven dataset (set4), with chronological separation (train up to 2010 and test with 2011 papers). These document embeddings where max pooled per dimention. We used 3 different K values to see how this affects metrics. 

### Data:
#### K values overlook:

| K value | 10 | 50 | 100 |
|:----|:----|:----|:----|
|Accuracy    |0.94915|0.95166|0.95057|
|Recall PS   |0.98428|0.98938|0.99008|
|Recall SR   |0.82382|0.81709|0.80962|
|Precision PS|0.95223|0.95073|0.94886|
|Precision SR|0.93628|0.95570|0.95812|

#### 10K ratios

|Predicted|Histogram(x: neighbors of predicted class, y: document count)|
|:--------|:--------|
|PS Right|![Histogram 10k right PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_right_10k.png)|
|PS Wrong|![Histogram 10k wrong PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_wrong_10k.png)|
|SR Right|![Histogram 10k right SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_right_10k.png)|
|SR Wrong|![Histogram 10k wrong SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_wrong_10k.png)|

|Ratio measure|Maximum|Minimun|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.90657|
|When predicted wrong PS|1.0|0.5|0.67044|
|When predicted right SR|1.0|0.6|0.90436|
|When predicted wrong SR|1.0|0.6|0.69600|

#### 50K ratios

|Predicted|Histogram (x: neighbors of predicted class, y: document count)|
|:--------|:--------|
|PS Right|![Histogram 50k right PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_right_50k.png)|
|PS Wrong|![Histogram 50k wrong PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_wrong_50k.png)|
|SR Right|![Histogram 50k right SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_right_50k.png)|
|SR Wrong|![Histogram 50k wrong SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_wrong_50k.png)|

|Ratio measure|Maximum|Minimun|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.88726|
|When predicted wrong PS|1.0|0.5|0.66267|
|When predicted right SR|1.0|0.52|0.86149|
|When predicted wrong SR|1.0|0.52|0.65789|


#### 100K ratio

|Predicted|Histogram (x: neighbors of predicted class, y: document count)|
|:--------|:--------|
|PS Right|![Histogram 100k right PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_right_100k.png)|
|PS Wrong|![Histogram 100k wrong PS](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/primary-study_histo_wrong_100k.png)|
|SR Right|![Histogram 100k right SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_right_100k.png)|
|SR Wrong|![Histogram 100k wrong SR](https://github.com/DiegoAndai/Deep-learning-framework-research/blob/master/Epistemonikos/ModelsEvaluation/PaperClassification/KNNClassification/Probability_study/systematic-review_histo_wrong_100k.png)|


|Ratio measure|Maximum|Minimum|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.87982|
|When predicted wrong PS|0.99|0.5|0.66497|
|When predicted right SR|1.0|0.51|0.84498|
|When predicted wrong SR|0.99|0.51|0.64894|

(Waiting for 150k and maybe 200k)!! update coming


### Analysis

These results confirm our hipothesis, the number of correct classifications rises when the prediction is based on a greater ratio. This means, first, that the classification method is working as expected on the premise that neighbour count is a good indicator for classification. Second, this means that in the high dimension space (D = 500) of the embeddings, is likely that decent defined clusters exist.

The good news is that this classifier could be useful as a guidance, as it can give a prediction with a logical confidence level. Could it classify on it's own without supervision? This is on of our main goals, to make an accurate classificator so we can reduce epistemonikos staff work load. We want, at least, an automated worker that is at least as good as a trained human one.

Note that the average ratio of correctly classified documents is higher, so for every neighbor of the predicted class, we are more certain of our decision. On the K = 100 case, the maximum ratio is 0.99 for both Primary Study ans Systematic Review wrong predictions, this means (for this particular documents) that there aren't any primary studies with all their neighbors being systematic reviews and vice versa. In other words, if all 100 neighbors are of class A, then statistically the document must be of class A, you could be certain of it. This is not the case of the other two classifications.

This brings up another an interrogant, what's most important, maximize certainty or correct classifications. On the premise that clusters exist, then it's logical to think that we can set some threshold ratio of certainty. On the 100k classifier, that threshold is 0.99, if you go up that number, there's statistical certainty (based on the observations) that we can classify correctly. If we can lower that threshold to, say, 0.9, the same classifier could have certainty for almost half of the entire test set.

This is a nice goal, that will reduce the work of epistemonikos staff to half without loosing accuracy. So how can we reduce the threshold is the question. One answer is to increase the K value, but that can reduce accuracy. Another answer is to eliminate outliers.


