import numpy as np
import sklearn.decomposition


class DimReducer:
    def __init__(self, dimLow=3):
        self.dimLow = dimLow

    def fit_transform(self, data, **kwargs):
        pass


class PCA(DimReducer):
    def __init__(self, dimLow):
        """ `PCA` is a class of `DimReducer`

        `PCA` is a statistical procedure that uses an orthogonal transformation
        to convert a set of observations of possibly correlated variables into
        a set of values of linearly uncorrelated variables called principal
        components.
        This class has a method fit_transform to call with the data to project
        as input.

        Args:
            dimLow (int): dimension of the low dimensional space
                          should be smaller than the number of colums of the
                          input data
        """

        if not isinstance(dimLow, int):
            raise TypeError("dimLow should be a positive integer")
        if dimLow < 1:
            raise ValueError("dimLow should be positive")
        self.dimLow = dimLow

        self._pca = sklearn.decomposition.PCA(n_components=self.dimLow)

    def fit(self, data, **kwargs):
        return self._pca.fit(data)

    def fit_transform(self, data, **kwargs):
        """`fit_transform` projects the input data on a lower dimensional space
        of dimension `dimLow`

        `fit_transform` reduces the number of columns of the input data to
        dimLow using an orthogonal transformation
        to convert a set of observations of possibly correlated variables into
        a set of values of linearly uncorrelated variables called principal
        components.
        This method calls the library Numpy

        Args:
            data (numpy.ndarray): input data that needs to be reduced.
                                  data should be a table of n lines being n
                                  observations, each line having p features.

        Returns:
            numpy.ndarray: reduced data, a table of n lines and `dimLow`
                           columns
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("Data should be a Numpy array")
        if len(data[0]) < self.dimLow:
            raise ValueError("Data has less columns than dimLow")

        reducedData = self._pca.fit_transform(data)
        return reducedData

    def transform(self, data):
        return self._pca.transform(data)


class ApproximatePCA(DimReducer):
    def __init__(
        self, dimLow, fracRow=0.01, fracCol=1.0, minRow=150, minCol=150
    ):
        """`ApproximatePCA` is a class of DimReducer

        ApproximatePCA is a randomized version of PCA. The first step is to
        randomly select rows and columns of the input matrix with probability
        proportionnal to their relative weight. Then to run the traditional
        principal component analysis. This method provides leading principal
        components very similar to exact PCA, but requires significantly less
        running time.

        Args:
            dimLow (int): dimension of the low dimensional space
                          should be smaller than the number of colums of the
                          input data
            fracRow (float<1): fraction of the number of rows to be used to
                               fit the data.
            fracCol (float<1): fraction of the number of columns to be used to
                               fit the data.
            minRow (int): minimum number of columns to be used to fit the data
            minCol (int): minimum number of rows to be used to fit the data
        """
        if not isinstance(dimLow, int):
            raise TypeError("dim Low should be an integer")
        if dimLow < 1:
            raise ValueError("dimLow should be positive")
        if not (isinstance(fracRow, float) or isinstance(fracRow, int)):
            raise TypeError("fracRow should be float")
        if fracRow <= 0 or fracRow > 1:
            raise ValueError("fracRow should be between 0 and 1")
        if not (isinstance(fracCol, float) or isinstance(fracCol, int)):
            raise TypeError("fracCol should be a float")
        if fracCol <= 0 or fracCol > 1:
            raise ValueError("fracCol should be between 0 and 1")
        if not isinstance(minRow, int):
            raise TypeError("minRow should be integer")
        if minRow <= 0:
            raise ValueError("minRow should be a positive integer")
        if not isinstance(minCol, int):
            raise TypeError("minCol should be an integer")
        if minCol <= 0:
            raise ValueError("minCol should be a positive integer")

        self.dimLow = dimLow
        self.fracRow = fracRow
        self.fracCol = fracCol
        self.minRow = minRow
        self.minCol = minCol

        self._pca = sklearn.decomposition.PCA(n_components=self.dimLow)

    def _get_proba_col(self, data):
        """
        returns a Numpy array of the probability to chose each collumn of data
        this probability is based on |c|**2/Froebenius
        input: numpy array
        output: numpy array
        """
        data = data.astype(float)
        result = sum(data ** 2)
        result /= sum(result)
        return result

    def _get_proba_row(self, data):
        """
        returns a Numpy array of the probability to chose each row of data
        this probability is based on |r|**2/Froebenius
        input: numpy array
        output: numpy array
        """
        data = data.astype(float)
        result = np.sum(data ** 2, axis=1)
        result /= sum(result)
        return result

    def _col_reduction(self, data):
        """
        Compute the probability for each collumn and reduce the number of
        collumns accordingly, keeping only minCol or fracCol*#col
        input: numpy array
        output: numpy array
        """
        proba_col = self._get_proba_col(data)
        n = len(data[0])
        n_col = max(self.minCol, n * self.fracCol, self.dimLow)
        n_col = int(min(n_col, len(data[0])))
        if n_col < n:
            list_col = np.random.choice(range(n), n_col, 0, proba_col)
            factor = np.sqrt(np.array(proba_col)[list_col] * n_col)
            result = np.copy(data[:, list_col]) / factor
            return result
        else:
            return data

    def _row_reduction(self, data):
        """
        Compute the probability for each row and reduce the number of
        rows accordingly, keeping only minRow or fracRow*#col
        input: numpy array
        output: numpy array
        """
        proba_row = self._get_proba_row(data)
        n = len(data)
        n_row = max(self.minRow, n * self.fracRow)
        n_row = int(min(n_row, len(data)))
        if n == n_row:
            return data
        else:
            list_rows = np.random.choice(range(0, n), n_row, 0, proba_row)
            result = np.copy(data[list_rows, :])
            return result

    def fit(self, data, seed=None, **kwargs):
        if abs(self.fracCol - 1.0) > 1e-8:
            raise NotImplementedError(
                "Fit & Transform methods are not implemented for column "
                "reduced data."
            )

        if seed:
            np.random.seed(seed)

        reduced_data = self._row_reduction(data)
        self._pca.fit(reduced_data)

    def fit_transform(self, data, seed=None, **kwargs):
        """`fit_transform` projects the input data on a lower dimensional space
        of dimension `dimLow`

        `fit_transform` reduces the number of columns of the input data to
        dimLow using an orthogonal transformation on a random subset of rows
        and columns to convert a set of observations of possibly correlated
        variables into a set of values of linearly uncorrelated variables
        called principal components.
        This method calls the library Numpy.

        Args:
            data (numpy.ndarray): input data that needs to be reduced.
                                  data should be a table of n lines being n
                                  observations, each line having p features.

        Returns:
            numpy.ndarray: reduced data, a table of n lines and `dimLow`
                           columns
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data should be a Numpy array")

        if abs(self.fracCol - 1.0) <= 1e-8:
            col_reduced_data = data
        else:
            col_reduced_data = self._col_reduction(data)

        if seed:
            np.random.seed(seed)

        reduced_data = self._row_reduction(col_reduced_data)
        self._pca.fit(reduced_data)
        return self._pca.transform(col_reduced_data)

    def transform(self, data, **kwargs):
        return self._pca.transform(data)
