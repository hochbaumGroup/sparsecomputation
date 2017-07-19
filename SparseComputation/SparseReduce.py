import numpy as np
import itertools


class SparseReduce:

    def __init__(self, dimReducer, gridResolution,  subblockResolution=1):
        '''`SparseReduce` is an object provided with a `DimReducer`,
        object and a `get_Reduced_data` method that returns a dcitionary of
        representative and a list of pairs of highly similar representative

        The sparse-reduced computation method works by first projecting the
        (high-dimensional) data set onto a low-dimensional space using a
        `DimReducer`. Then grid blocks are created. Each block is then
        subdivided in subblocks representing several datapoints. Then we use
        grid neighborhood to select the pair of representative that are deemed
        to be highly similar.

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
            subblockResolution (int): number of subblocks per axis per block.
                                      For instance, a `DimReducer` provided
                                      with a `dimLow` of 3 and a
                                      `SparseComputation` provided with a
                                      `gridResolution` of 4 will build 4^3=64
                                      grid blocks
        '''
        if not isinstance(gridResolution, int):
            raise TypeError('gridResolution should be a positive integer')
        if gridResolution < 1:
            raise ValueError('gridResolution should be a positive integer')
        if subblockResolution < 1:
            raise ValueError('subblockResolution should be a positive integer')

        self.dimReducer = dimReducer
        self.gridResolution = gridResolution
        self.subblockResolution = subblockResolution

    def _rescale_data(self, data):
        '''
        Rescale the data from 0 to gridResolution*subblockResolution-1
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

        Resolution = self.gridResolution*self.subblockResolution

        coef = float(Resolution) / gap

        rescaled_data = (data - minimum) * coef
        rescaled_data = rescaled_data.astype('int')

        rescaled_data = np.where(rescaled_data < Resolution,
                                 rescaled_data, Resolution - 1)
        return rescaled_data

    def _get_sub_blocks(self, data):
        '''
        get the data after rescaling
        sort it in subblocks
        input: np.array
        output: dict of subblocks
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

    def _get_boxes(self, subblocks, data):
        '''
        get the data after rescaling and the dictionary of subblocks
        sort it in blocks
        input: np.array
        output: dict of subblocks
        '''
        n = len(subblocks)
        result = {}
        for key in subblocks.keys():
            box_id = data[subblocks[key][0]]/self.subblockResolution
            box_id = np.ndarray.astype(box_id, int)
            box_id = tuple(box_id)
            if not (box_id in result):
                result[box_id] = []
            result[box_id].append(key)
        return result

    def _get_pairs(self, data, boxes_dict):
        '''
        get reduced dimension data returns a list of one way pairs for
        similarities to be computed
        input: np.array
        output: list of pairs of subblocks
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError

        n = len(data[0])
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

    def get_Reduced_data(data):
        '''
        `get_Reduced_data` is a method that first projects `data` onto a low
        dimensional space using the provided `DimReducer`
        It then groups delta-identical datapoints into a common representative.
        Then grid blocks are created and we use grid neighborhood to select the
        pairs of representative that are deemed to be highly similar.

        Args:
            data (numpy.ndarray): input data that needs to be spasified
                                  data should be a table of n lines being n
                                  observations, each line having p features.
        Returns:
            (dict {(int,..): [int]}): dictionary of representative. Each key is
                                      a representative id, the list linked to
                                      it is the list of the datapoint (rows of
                                      data) representated by it.
            (list [((int,..), (int,..))]): list of undirected pairs of
                                           representative. This mean that if
                                           `i` is similar to `j` both pairs
                                           `(i, j)` and `(j, i)` are returned
                                           in the list
        '''
        reduced_data = self.dimReducer.fit_transform(data)
        rescaled_data = self._rescale_data(reduced_data)
        subblocks = self._get_sub_blocks(rescaled_data)
        blocks = self._get_boxes(subblocks, rescaled_data)
        pairs = self._get_pairs(rescaled_data, blocks)
        return (subblocks, pairs)
