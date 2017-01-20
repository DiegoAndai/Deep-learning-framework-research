## Probability Study for KNN Classification

With this, we want to analyze how the classifier's confidence behaves in relation to it's right/wrong rate. Here, confidence is measured in number of neighbors of the predicted class. Our hipothesis is, as the ratio of neighbors becomes greater, the classifier should get more predictions right. 

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

#### 10K probabilities

![Histogram 10k right]()
![Histogram 10k wrong]()

|Probability measure|Maximum|Minimun|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.90657|
|When predicted wrong PS|1.0|0.5|0.67044|
|When predicted right SR|1.0|0.6|0.90436|
|When predicted wrong SR|1.0|0.6|0.69600|

#### 50K probabilities

![Histogram 50k right]()
![Histogram 50k wrong]()

|Probability measure|Maximum|Minimun|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.88726|
|When predicted wrong PS|1.0|0.5|0.66267|
|When predicted right SR|1.0|0.52|0.86149|
|When predicted wrong SR|1.0|0.52|0.65789|


#### 100K probabilities

![Histogram 100k right]()
![Histogram 100k wrong]()

|Probability measure|Maximum|Minimun|Average|
|:------------------|:------|:------|:------|
|When predicted right PS|1.0|0.5|0.87982|
|When predicted wrong PS|0.99|0.5|0.66497|
|When predicted right SR|1.0|0.51|0.84498|
|When predicted wrong SR|0.99|0.51|0.64894|


# 
