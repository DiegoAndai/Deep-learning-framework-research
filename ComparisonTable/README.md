##Comparison table for neural network frameworks

| Name      | Complexity | Performance | Interface | Used in*(or comments?) | Developing OS  | 
| :-------  | :--------- | :----------------- | :---------- | :-------- | :------------- | :-------    |
| [Caffe]() |b, c|                    |             |Fast R-CNN, Expresso| CentOS, Fedora, Mac OS X, RHEL, Ubuntu, Windows. (Docker, OpenCL) |
| [Keras]() |b, c|                    |             |DeepJazz, visua-qa| CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. |  
| [Torch]() |a, b, c|                    |             |Neural-style, NeuralTalk2| Mac OS X, Ubuntu. (Docker, AWS) *¿CentOS, RHEL? | 
| [Tensorflow]() |a|                    |             |RankBrain, Keras| Mac OS X, Ubuntu    |
| [CNTK]() |b|         | |Microsoft cognitive services|Ubuntu, Windows. *¿+Linux? | 
| [Theano]() |a|                    |             |Keras, Lasagne| CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. (Docker) | 
| [Lasagne]() |b|                    |             |NTM, Lasagne-draw| CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. (Docker)|  
| [Mxnet]() |a*|                    |             |Neural-art-mini, Mxnet-face| Mac OS X, Ubuntu, Windows. |  

#Complexity indices:
  
    a: math or node tools to implement flow graphs
    b: layer tools to implement neural networks, including node customization
    c: implemented models ready to use
  
    *having a higher complexity level doesn't mean that the lower levels can also be used. 
