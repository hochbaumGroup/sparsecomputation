import numpy as np
import itertools
from six.moves import zip


class SparseComputation:

    def __init__(self, dimReducer, gridResolution=25):
        '''`SparseComputation` is an object provided with a `DimReducer`
        object and a `get_similar_indices` method that returns the highly
        similar pairs

        The sparse computation method works by first projecting the
        (high-dimensional) data set onto a low-dimensional space using a
        `DimReducer`. Then grid blocks are created and we use grid neighborhood
        to select the pairs that are deemed to be highly similar.

        Args:
            dimReducer (DimReducer): object of the class `DimReducer` provided
                                     with a `fit_transform` method to project
                                     the data on a smaller dimensional space
                                     before the similar pairs are computed
            gridResolution (int): number of gridblock per axis. For instance,
                                  a `DimReducer` provided with a `dimLow` of 3
                                  and a `SparseComputation` provided with a
                                  `gridResolution` of 4 will build 4^3=64
                                  grid blocks
        '''
        if not isinstance(gridResolution, int):
            raise TypeError('gridResolution should be a positive integer')
        if gridResolution < 1:
            raise ValueError('gridResolution should be a positive integer')

        self.dimReducer = dimReducer
        self.gridResolution = gridResolution

    def _rescale_data(self, data):
        '''
        Rescale the data from 0 to gridResolution-1
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('Data should be a Numpy array')

        n = len(data[0])
        maximum = np.amax(data, axis=0, keepdims=True)
        minimum = np.amin(data, axis=0, keepdims=True)

        gap = maximum - minimum
        gap = np.where(gap > 0, gap, 1)

        coef = float(self.gridResolution) / gap

        rescaled_data = (data - minimum) * coef
        rescaled_data = rescaled_data.astype('int')

        rescaled_data = np.where(rescaled_data < self.gridResolution,
                                 rescaled_data, self.gridResolution - 1)

        return rescaled_data

    def _get_boxes(self, data):
        '''
        get the data after rescaling
        sort it in boxes
        input: np.array
        output: dict of boxes
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('Data should be a Numpy array')

        n = len(data[0])
        result = {}
        for i in range(0, len(data)):
            box_id = tuple(data[i])
            if not (box_id in result):
                result[box_id] = []
            result[box_id].append(i)
        return result

    def _get_pairs(self, data):
        '''
        get reduced dimension data returns a list of one way pairs for
        similarities to be computed
        input: np.array
        output: list of pairs
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError

        n = len(data[0])
        rescaled_data = self._rescale_data(data)
        boxes_dict = self._get_boxes(rescaled_data)
        pairs = []

        increments = tuple(increment
                           for increment
                           in itertools.product(range(-1, 2), repeat=n)
                           if increment > ((0, ) * n)
                           )

        for box_id in boxes_dict:
            pairs += itertools.combinations(boxes_dict[box_id], 2)
            for increment in increments:
                id_incremented = tuple(a + b
                                       for a, b
                                       in zip(box_id, increment))
                if id_incremented in boxes_dict:
                    pairs += itertools.product(
                        boxes_dict[box_id], boxes_dict[id_incremented]
                        )
        return pairs

    def get_similar_indices(self, data, seed=None, **kwargs):
        '''`get_similar_indices` computes the similar indices in the data and
        return a list of pairs

        `get_similar_indices` first projects the (high-dimensional) data set
        onto a low-dimensional space using the `DimReducer`. Then grid blocks
        are created and we use grid neighborhood to select the pairs that are
        deemed to be highly similar.

        Args:
            data (numpy.ndarray): input data that needs to be spasified
                                  data should be a table of n lines being n
                                  observations, each line having p features.
        Returns:
            (list [(int, int)]): list of directed pairs. This mean that if `i`
                                 is similar to `j` both pairs `(i, j)` and
                                 `(j, i)` are returned in the list
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('data should be a numpy array')

        reduced_data = self.dimReducer.fit_transform(data, seed)
        return self._get_pairs(reduced_data)
