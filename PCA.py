from DimReducer import DimReducer
import numpy as np
from sklearn.decomposition import PCA

class PCA ( DimReducer ):
    def __init__ (self, dimLow):
        if not isinstance(dimLow, int):
            raise TypeError
        if dimLow<2:
            raise ValueError
        self.dimLow = dimLow

    def fit_transform (self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])<self.dimLow:
            raise ValueError
        pca = PCA(n_components=self.dimLow, svd_solver='full')
        reducedData = pca.fit_transform(data)
        return reducedData
