![Travis build status](https://travis-ci.org/hochbaumGroup/sparsecomputation.svg?branch=master)

# Sparse Computation
`sparsecomputation` is a python package to construct a sparse matrix of pairwise similarities (similarity matrix). Sparse similarity matrices generated with sparse computation provide a substantial speedup without loss in accuracy for supervised machine learning algorithms that rely on pairwise similarity matrices / kernel matrices, such as k-nearest neighbors, kernel-based support vector machines, and supervised normalized cut.

Similarity matrices have `O(n^2)` entries for  a dataset with `n` objects (observations). Computing all pairwise similarities is computationally intractable for large datasets. Sparse computation overcomes this computational burden by identifying all pairs of objects that are close to each other in space. These pairs are identified by projecting the objects onto a low-dimensional space and using closeness in the low-dimensional space as a proxy for closeness in the original space. Sparse computation uses grids to determine efficiently close objects in the low-dimensional space.

Sparse computation takes as input a two-dimensional numpy array of size `n x d`. Each row represents one object with `d` features. Spase computation returns a list of 2-tuples, where each tuple are the indices of the objects between which a pairwise similarity should be computed.

## Installation
You can use `pip` to install the package:
```
pip instal sparsecomputation
```

## Minimal Example
To find the relevant pairwise similarities with Sparse Computation, use the following code:
```python
# load sample dataset
import sklearn.datasets
data, _ = sklearn.datasets.load_iris(return_X_y=True)

# load Sparse Computation
from sparsecomputation import ApproximatePCA, SparseComputation

# Project data to a low-dimensional space of dimension 3
# with a fast, scalable version of PCA.
apca = ApproximatePCA(dimLow=3)
# Select pair if objects are closer than ~distance~ (1 / resolution).
# Controls sparsity (small distance - highly sparse matrix)
sc = SparseComputation(apca, distance=0.05)

# Input: n x d matrix - data
# Output: list of relevant pairs (undirected)
pairs = sc.select_pairs(data)
# Out: [(101, 142), (1, 9), (1, 34), (1, 37), ...]
```

## Relevant Papers
For more details on the techniques read the following papers. Please cite these works if you use this implementation in academic work:
- Dorit S. Hochbaum, Philipp Baumann (2016). Sparse computation for large-scale data mining. *IEEE Transactions on Big Data*, 2(2), 151-174.
- Philipp Baumann, Dorit S. Hochbaum and Quico Spaen (2017). High-Performance Geometric Algorithms for Sparse Computation in Big Data Analytics. *2017 IEEE International Conference on Big Data*, Boston MA.
- Philipp Baumann, Dorit S. Hochbaum, and Quico Spaen (2016). Sparse-Reduced Computation. *Proceedings of the 5th International Conference on Pattern Recognition Applications and Methods*.

## Credits
Original Matlab implementation by Philipp Baumann. Original Python implementation by Titouan Jehl and Quico Spaen. Updated implementation with the new block shifting algorithm by Quico Spaen and Philipp Baumann.
