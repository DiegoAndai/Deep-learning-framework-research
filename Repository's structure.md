## Why is this necesary?

There are some large datasets needed to train models that are not on this repository, if you want to use code from it with those databases or maybe with others, here we list how we structured our files to prevent annoying buggy programs. 

## Structure
```
├──Readme.md
├──Repository's structure.md
├──ComparisonTable
│   ├── README.md
│   ├── categories.md
│   ├── Install_reports
│       ├── Caffe.md
│       ├── Keras.md
│       ├── Lasagne.md
│       ├── Mxnet.md│
│       ├── Tensorflow.md
│       ├── Theano.md
│       ├── Torch.md
│       ├── install_report_format.md
│       ├── Install_tests
│           ├── Mxnet install test.ipynb
│           ├── mnist_parse.py
│           ├── mnist
│               ├── loader.py  
│
├──Epistemonikos
│   ├── Analogy
│   │   ├── analogy.py
│   │   ├── analogy_pairs.py
│   │
│   ├── PaperClassification
│   │   ├── classify.py
│   │
│   ├── Relatedness
│   │   ├── UMNSRS_relatedness.csv
│   │   ├── relatedness.py
│   │   ├── wordsim_relatedness_goldstandard.txt
│   │
│   ├── Skipgram
│       ├── embedding
│       ├── embedding_for_episte.py
│       ├── open_documents.py
│       ├── reverse_dictionary
│       ├── visualization.py
│       ├── Advanced
│           ├── word2vec_optimized.py
│
├── MNIST
│   ├── README.md
│   ├── Keras
│   │   ├── Keras_mnist_mlp.ipynb
│   │
│   ├── Mxnet
│   │   ├── Mxnet_mnist_mlp.ipynb
│   │
│   ├── Tensorflow
│   │   ├── tf.py
│   │   ├── tf.png
│   │
│   ├── Torch
│   │   ├── torch.png
│   │   ├── torch.lua
│   │   ├── MNISTParser.lua
│   │
│   ├── mnist_data_loader
│       ├── loader.py
│       ├── mnist_parse.py
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│
├── SOWebsite
│   ├── Labs.py
│   ├── Plot.py
│   ├── Run.py
│   ├── static
│   │   ├── eztylo.css
│   │   ├── plots
│   │       ├── overview_plot.png
│   │
│   ├── templates
│       ├── base.html
│       ├── framework.html
│           ├── overview.html
│   
├── Word2Vec
    ├── 3 datasets word embedding 1,000.png
    ├── 3 datasets word embedding 10,000.png
    ├── 3 datasets word embedding 2,500.png
    ├── 3 datasets word embedding 5,000.png
    ├── Word2Vec.ipynb
    ├── movie quotes word embedding 1,000.png
    ├── movie quotes word embedding 10,000.png
    ├── movie quotes word embedding 5,000.png

```


