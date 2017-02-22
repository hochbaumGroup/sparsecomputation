from DimReducer import DimReducer
import numpy as np
import sklearn.decomposition


class ApproximatePCA(DimReducer):

    def __init__(self, dimLow, percRow=0.01,
                 percCol=1, minRow=150, minCol=150):
        if not isinstance(dimLow, int):
            return TypeError
        if dimLow > 3:
            return ValueError
        if not (isinstance(percRow, float) or isinstance(percRow, int)):
            return TypeError
        if percRow <= 0 or percRow > 100:
            return ValueError
        if not (isinstance(percCol, float) or isinstance(percCol, int)):
            return TypeError
        if percCol <= 0 or percCol > 100:
            return ValueError
        if not isinstance(minRow, int):
            return TypeError
        if minRow <= 0:
            return ValueError
        if not isinstance(minCol, int):
            return TypeError
        if minCol <= 0:
            return ValueError
        self.dimLow = dimLow
        self.percRow = percRow
        self.percCol = percCol
        self.minRow = minRow
        self.minCol = minCol

    def _get_Frobenius(self, data):
        if not isinstance(data, np.ndarray):
            return TypeError
        result = 0
        for vec in data:
            result += float(sum(vec**2))
        return result

    def _get_proba_col(self, data):
        if not isinstance(data, np.ndarray):
            return TypeError
        Frobenius = self._get_Frobenius(data)
        transposed_data = np.transpose(data)
        proba_col = []
        for col in transposed_data:
            proba_col.append(sum(col**2)/Frobenius)
        return proba_col

    def _get_proba_row(self, data):
        if not isinstance(data, np.ndarray):
            return TypeError
        Frobenius = self._get_Frobenius(data)
        proba_row = []
        for row in data:
            proba_row.append(sum(row**2)/Frobenius)
        return proba_row

    def _col_reduction(self, data):
        if not isinstance(data, np.ndarray):
            return TypeError

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
        if not isinstance(data, np.ndarray):
            return TypeError
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
        if not isinstance(data, np.ndarray):
            return TypeError
        col_reduced_data = self._col_reduction(data)
        reduced_data = self._row_reduction(col_reduced_data)
        pca = sklearn.decomposition.PCA(n_components=self.dimLow,
                                        svd_solver='full')
        pca.fit(reduced_data)
        data = pca.transform(col_reduced_data)
        return data
