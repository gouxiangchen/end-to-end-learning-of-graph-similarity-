# end-to-end-learning-of-graph-similarity-
The test code and trained model for paper "End-to-End Learning of Graph Similarity"

The dataset is quite large and some samples are available for test

# notice
The version is poor implementation and not suitable for everyone, I will enrich the doc and reimplement the code later. The network data is collected from KONECT (http://konect.uni-koblenz.de/) and some augmentation techniques are used to enlarge networks datasets, which are introduced in our paper. 

The train code is not available now. They need much more modification to be public.

# requirements 

+ numpy
+ torch (1.0)
+ networkx

# usage
modify the filenames in test_main.py & Graphlet.py to change the pair of graph you want to test, and just run 

```
python test_main.py

python Graphlet.py
```
to get the model prediction and the label distance, the gap between prediction and the label is the error of our model. 

# thanks

The graphlets counting method is borrowed from liuxt at [liuxt's graphlets counting method](https://github.com/liuxt/graphlet_counting)


