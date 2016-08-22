##Comparison table for neural network frameworks

| Name      | Complexity | Performance | Interface | Used in        | Developing OS*check  | Comments |
| :-------  | :--------- | :----------------- | :---------- | :-------- | :------------- | :-------    |:------------- |
| [Caffe]() |b, c|                    |             |           | CentOS, Fedora, Mac OS X, RHEL, Ubuntu, Windows. (Docker, OpenCL) | |
| [Keras]() |b, c|                    |             |           | CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. |  |
| [Torch]() |a, b, c|                    |             |           | Mac OS X, Ubuntu. (Docker, AWS) *¿CentOS, RHEL? |  |
| [Tensorflow]() |a|                    |             |           | Mac OS X, Ubuntu    |         |
| [CNTK]() |a, b|         | | |Ubuntu, Windows. *¿+Linux? | |
| [Theano]() |a|                    |             |           | CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. (Docker) |  |
| [Lasagne]() |b|                    |             |           | CentOS, Gentoo, Mac OS X, Mandriva, Ubuntu, Windows. (Docker)|  |
| [Mxnet]() |a*|                    |             |           | Mac OS X, Ubuntu, Windows. |   |

#Complexity indices:
  
    a: math or node tools to implement flow graphs
    b: layer tools to implement neural networks
    c: implemented models ready to use
  
    *having a higher complexity level doesn't mean that the lower levels can also be used. 
