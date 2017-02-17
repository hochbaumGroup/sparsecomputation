import numpy as np
import itertools


class SparseComputation:

    def __init__(self, dimReducer, gridResolution=25):
        if not isinstance(gridResolution, int):
            raise TypeError
        if gridResolution < 1:
            raise ValueError
        self.dimReducer = dimReducer
        self.gridResolution = gridResolution

    def _get_min(self, data):
        '''
        take the data and return the minimum for each dimension in a list
        input: numpy array
        output: list of length dimLow
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0]) > 3:
            raise ValueError
        n = len(data[0])
        minimum = np.copy(data[0])
        for vec in data:
            for i in range(0, n):
                if vec[i] < minimum[i]:
                    minimum[i] = vec[i]
        return minimum

    def _get_max(self, data):
        '''
        take the data and return the max for each dimension in a list
        input: numpy array
        output: list of length dimLow
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0]) > 3:
            raise ValueError
        n = len(data[0])
        maximum = np.copy(data[0])
        for vec in data:
            for i in range(0, n):
                if vec[i] > maximum[i]:
                    maximum[i] = vec[i]
        return maximum

    def _rescale_data(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0]) > 3:
            raise ValueError
        n = len(data[0])
        maximum = self._get_max(data)
        minimum = self._get_min(data)
        coef = []
        rescaled_data = []
        for i in range(0, n):
            try:
                toAppend = float(self.gridResolution)/(maximum[i]-minimum[i])
                coef.append(toAppend)
            except ZeroDivisionError:
                coef.append(0)
        for vec in data:
            rescaled_vec = []
            for i in range(0, n):
                rescaled_vec.append(int((vec[i]-minimum[i])*coef[i]))
            rescaled_data.append(rescaled_vec)
        return np.array(rescaled_data)

    def _index_to_boxe_id(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError
        if len(array) > 3:
            raise ValueError
        if not isinstance(array[0], int):
            raise TypeError
        result = 0
        for i in range(0, len(array)):
            result += array[i]*self.gridResolution**i
        return result

    def _get_boxes(self, data):
        '''
        get the data after rescaling
        sort it in boxes
        input: np.array
        output: dict of boxes
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0]) > 3:
            raise ValueError
        n = len(data[0])
        result = {}
        for i in range(0, len(data)):
            box_id = self._index_to_boxe_id(data[i])
            if not (box_id in result):
                result[box_id] = []
            result[box_id].append(i)
        return result

    def get_sim_indices(self, data):
        '''
        get reduced dimension data returns a list of one way pairs for
        similarities to be computed
        input: np.array
        output: list of pairs
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0]) > 3:
            raise ValueError
        n = len(data[0])
        rescaled_data = self._rescale_data(data)
        boxes_dict = self._get_boxes(rescaled_data)
        pairs = []
        for box_id in boxes_dict:
            for j in itertools.product(boxes_dict[box_id], boxes_dict[box_id]):
                pairs.append(j)
            for i in range(0, n):
                id_plus = box_id + self.gridResolution**i
                id_minus = box_id - self.gridResolution**i
                if id_plus in boxes_dict:
                    for j in itertools.product(boxes_dict[box_id], boxes_dict[id_plus]):
                        pairs.append(j)
                if id_minus in boxes_dict:
                    for j in itertools.product(boxes_dict[box_id], boxes_dict[id_minus]):
                        pairs.append(j)
        return pairs
