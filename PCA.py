from DimReducer import DimReducer
import numpy as np
import sklearn.decomposition


class PCA (DimReducer):
    def __init__(self, dimLow):
        if not isinstance(dimLow, int):
            raise TypeError('dimLow should be a positive integer')
        if dimLow < 2 or dimLow > 3:
            raise ValueError('dimLow should be 2 or 3')
        self.dimLow = dimLow

    def fit_transform(self, data):
        '''
        Usual fit_transform of numpy.decomposition.PCA
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('Data should be a Numpy array')
        if len(data[0]) < self.dimLow:
            raise ValueError('Data has less collumns than dimLow')

        pca = sklearn.decomposition.PCA(n_components=self.dimLow,
                                        svd_solver='full')
        reducedData = pca.fit_transform(data)
        return reducedData
