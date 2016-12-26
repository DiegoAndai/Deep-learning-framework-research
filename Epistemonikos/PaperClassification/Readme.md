This folder is intended to contain different ways to classify papers.
Currently the following techniques have been implemented:

* PVPClassification (pondered vector paper classification): obtains 
1 representative vector for each paper class, another vector for each
paper to be classified and compares the vector of each paper with
the vectors of the paper classes, assigning to the papers the class that
has the most similar vector (in terms of orientation).
