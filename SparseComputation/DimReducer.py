import numpy as np
import sklearn.decomposition


class DimReducer:

    def __init__(self, dimLow=3):
        self.dimLow = dimLow

    def fit_transform(self, data):
        pass


class PCA (DimReducer):
    def __init__(self, dimLow):
        if not isinstance(dimLow, int):
            raise TypeError('dimLow should be a positive integer')
        if dimLow < 1:
            raise ValueError('dimLow should be positive')
        self.dimLow = dimLow

    def fit_transform(self, data):
        '''
        Data is a numpy array containing your data.
        Each row should be an observation and each column correspond to a
        parameter.
        fit_transform calls the PCA reduction of numpy library.
        The output will contain exactly the same number of rows but only
        dimLow columns.
        PCA projects the matrix data in a space of dimLow dimension minimizing
        the square error between the projection and the original vector.
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('Data should be a Numpy array')
        if len(data[0]) < self.dimLow:
            raise ValueError('Data has less columns than dimLow')

        pca = sklearn.decomposition.PCA(n_components=self.dimLow)
        reducedData = pca.fit_transform(data)
        return reducedData


class ApproximatePCA(DimReducer):

    def __init__(self, dimLow, percRow=0.01,
                 percCol=1.0, minRow=150, minCol=150):
        if not isinstance(dimLow, int):
            return TypeError('dim Low should be an integer')
        if dimLow < 1:
            return ValueError('dimLow should be positive')
        if not (isinstance(percRow, float) or isinstance(percRow, int)):
            return TypeError('percRow should be float')
        if percRow <= 0 or percRow > 1:
            return ValueError('percRow should be between 0 and 1')
        if not (isinstance(percCol, float) or isinstance(percCol, int)):
            return TypeError('percCol should be a float')
        if percCol <= 0 or percCol > 1:
            return ValueError('percCol should be between 0 and 1')
        if not isinstance(minRow, int):
            return TypeError('minRow should be integer')
        if minRow <= 0:
            return ValueError('minRow should be a positive integer')
        if not isinstance(minCol, int):
            return TypeError('minCol should be an integer')
        if minCol <= 0:
            return ValueError('minCol should be a positive integer')

        self.dimLow = dimLow
        self.percRow = percRow
        self.percCol = percCol
        self.minRow = minRow
        self.minCol = minCol

    def _get_proba_col(self, data):
        '''
        returns a Numpy array of the probability to chose each collumn of data
        this probability is based on |c|**2/Froebenius
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        data = (data.astype(float))
        result = sum(data ** 2)
        result /= sum(result)
        return result

    def _get_proba_row(self, data):
        '''
        returns a Numpy array of the probability to chose each row of data
        this probability is based on |r|**2/Froebenius
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        data = data.astype(float)
        result = np.sum(data**2, axis=1)
        result /= sum(result)
        return result

    def _col_reduction(self, data):
        '''
        Compute the probability for each collumn and reduce the number of
        collumns accordingly, keeping only minCol or percCol*#col
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        proba_col = self._get_proba_col(data)
        n = len(data[0])
        n_col = max(self.minCol, n*self.percCol / 100.0)
        list_col = np.random.choice(range(n), n_col, 0, proba_col)
        result = np.copy(data[:, list_col])
        return result

    def _row_reduction(self, data):
        '''
        Compute the probability for each row and reduce the number of
        rows accordingly, keeping only minRow or percRow*#col
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        proba_row = self._get_proba_row(data)
        n = len(data)
        n_row = max(self.minRow, n*self.percRow/100.0)
        list_rows = np.random.choice(range(0, n), n_row, 0, proba_row)
        result = np.copy(data[list_rows, :])
        return result

    def fit_transform(self, data):
        '''
        Apply approximate PCA to data and return the reduced data
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        col_reduced_data = self._col_reduction(data)
        reduced_data = self._row_reduction(col_reduced_data)
        pca = sklearn.decomposition.PCA(n_components=self.dimLow)
        pca.fit(reduced_data)
        data = pca.transform(col_reduced_data)
        return data
