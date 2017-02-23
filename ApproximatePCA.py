from DimReducer import DimReducer
import numpy as np
import sklearn.decomposition


class ApproximatePCA(DimReducer):

    def __init__(self, dimLow, percRow=0.01,
                 percCol=1, minRow=150, minCol=150):
        if not isinstance(dimLow, int):
            return TypeError('dim Low should be an integer')
        if dimLow > 3 or dimLow < 2:
            return ValueError('dimLow should be 2 or 3')
        if not (isinstance(percRow, float) or isinstance(percRow, int)):
            return TypeError('percRow should be float or integer')
        if percRow <= 0 or percRow > 100:
            return ValueError('percRow should be between 0 and 100')
        if not (isinstance(percCol, float) or isinstance(percCol, int)):
            return TypeError('percCol should be a float or an integer')
        if percCol <= 0 or percCol > 100:
            return ValueError('percCol should be between 0 and 100')
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

    def _get_Frobenius(self, data):
        '''
        return the Froebenius norm of the matrix data
        input: numpy array
        output: float
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        result = 0
        for vec in data:
            result += float(sum(vec**2))
        return result

    def _get_proba_col(self, data):
        '''
        returns a Numpy array of the probability to chose each collumn of data
        this probability is based on |c|**2/Froebenius
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        Frobenius = self._get_Frobenius(data)
        transposed_data = np.transpose(data)
        proba_col = []
        for col in transposed_data:
            proba_col.append(sum(col**2)/Frobenius)
        return np.array(proba_col)

    def _get_proba_row(self, data):
        '''
        returns a Numpy array of the probability to chose each row of data
        this probability is based on |r|**2/Froebenius
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            return TypeError('Data should be a Numpy array')

        Frobenius = self._get_Frobenius(data)
        proba_row = []
        for row in data:
            proba_row.append(sum(row**2)/Frobenius)
        return np.array(proba_row)

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
        if n != len(proba_col):
            raise TypeError
        list_col = np.random.choice(range(n), n_col, 0, proba_col)
        result = []
        transposed_data = np.transpose(data)
        for idx in list_col:
            result.append(np.copy(transposed_data[idx]))
        result = np.array(result)
        return np.transpose(result)

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
        if n != len(proba_row):
            raise TypeError
        n_row = max(self.minRow, n*self.percRow/100.0)
        list_rows = np.random.choice(range(0, n), n_row, 0, proba_row)
        result = []
        for idx in list_rows:
            result.append(np.copy(data[idx]))
        return np.array(result)

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
        pca = sklearn.decomposition.PCA(n_components=self.dimLow,
                                        svd_solver='full')
        pca.fit(reduced_data)
        data = pca.transform(col_reduced_data)
        return data
