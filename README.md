# sparseComputation

SparseComputation is a python package to construct a sparse similarity matrix from inputs in the form of pairwise similarities.

The purpose is to overcome thecomputational burden of computing all pairwise similarities between the data points by generating only the relevant similarities. Once the relevant pairs of objects have been identified, their similarities can be computed in the original space.

The input of data must be a numpy array of n observations, each observations presenting p features. The sparse computation method works by first projecting the (high-dimensional) data set onto a low-dimensional space using a `DimReducer`. Then grid blocks are created and we use grid neighborhood to select the pairs that are deemed to be highly similar.

## DimReducer

We first need to create an object of the class `DimReducer`. By definition all object of this class have a `dimLow` variable and a fit_transform function. Its goal is to project the high dimensional data onto a low dimensional space, reducing the number of columns to `dimLow`.

We propose two `DimReducer`: PCA and ApproximatePCA. Those are two statistical procedure that use an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

### Reduce dimension with principal component analysis (PCA)

Let us start with creating a PCA object of `dimLow` = 3

    pca = PCA(3)

Our `DimReducer` is created. Let data be a numpy array (n, p)

    pca.fit_transform(data) # returns a numpy array (n, 3)

### Reduce dimension with ApproximatePCA

ApproximatePCA is a randomized version of PCA. The first step is to randomly select rows and columns of the input matrix with probability proportionnal to their relative weight. Then to run the traditional principal component analysis. This method provides leading principal components very similar to exact PCA, but requires significantly less running time.

For this `DimReducer`, we need to specify what fraction of the total number of rows and columns need to be used to run PCA on as well as the minimum number.

    approximatepca = ApproximatePCA(dimLow = 3, fracRow=0.01, fracCol=0.1, minRow=150, minCol=150)

In this example, we will run PCA with `dimLow` = 3 on a reduced matrix containing 1% of the rows and 10% of the columns with a minimum of 150 for each.

Then similarily for data a numpy array (n, p)

    approxpca.fit_transform(data) # returns a numpy array (n, 3)

## Get similar indices

Let us create a SparseComputation object with a regular principal component analysis as `DimReducer` that will create 25 grid blocks per dimensions.

    pca = PCA(dimLow=3)
    sc = SparseComputation(dimReducer=pca, gridResolution=25)

Here, the high dimensional data is projected on a 3D space, then for each of the three dimensions, we split the space in 25 blocks. We have total 25^3 blocks. Finally `get_similar_indices` uses grid neighborhood to return the pairs of similar indices. Those pairs are directed.

For data a numpy array (n, p)

    sc.get_similar_indices(data) # returns a list of directed pairs
