import itertools
import six.moves
import numpy as np
from sklearn import preprocessing


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
        self.intervalLength = 1/ float(self.gridResolution)

    def _rescale_data(self, data):
        '''
        Rescale the data from 0 to gridResolution-1
        input: numpy array
        output: numpy array
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('Data should be a Numpy array')

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

        result = {}
        for i in range(0, len(data)):
            box_id = tuple(data[i])
            if not (box_id in result):
                result[box_id] = []
            result[box_id].append(i)
        return result

    def _get_pairs(self, data, statistics=False):
        '''
        get reduced dimension data returns a list of one way pairs for
        similarities to be computed
        input: np.array
        output: list of pairs
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError

        if statistics:
            statistics_dict = {}
            num_boxes = 0
            num_neighbors = 0
            num_nonempty_neighbors = 0

        n = len(data[0])
        rescaled_data = self._rescale_data(data)
        boxes_dict = self._get_boxes(rescaled_data)
        if statistics:
            num_boxes = len(boxes_dict)
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
                                       in six.moves.zip(box_id, increment))
                if statistics:
                    num_neighbors += 1
                if id_incremented in boxes_dict:
                    if statistics:
                        num_nonempty_neighbors += 1
                    pairs += itertools.product(
                        boxes_dict[box_id], boxes_dict[id_incremented]
                        )
        output = {'pairs': pairs}
        if statistics:
            statistics_dict['num_boxes'] = num_boxes
            statistics_dict['num_neighbors'] = num_neighbors
            statistics_dict['num_nonempty_neighbors'] = num_nonempty_neighbors
            statistics_dict['num_empty_neighbors'] = num_neighbors-num_nonempty_neighbors
            output['statistics'] = statistics_dict
        return output

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

        Keyword arguments:
            statistics (bool):    if true, dictionary with statistics is added
                                  to output

        Returns:
            dictionary:          contains list of pairs and (if statistics
                                 is true) also a dictionary of statistics
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError('data should be a numpy array')

        if 'statistics' in [key for key in kwargs]:
            statistics = kwargs['statistics']
        else:
            statistics = False

        if self.dimReducer is None:
            reduced_data = data
        else:
            reduced_data = self.dimReducer.fit_transform(data, seed=seed)
        return self._get_pairs(reduced_data, statistics=statistics)

class SparseShiftedComputation(SparseComputation):
    '''
    Alternative technique to SparseComputation that avoids identifying
    neighboring blocks
    '''

    @staticmethod
    def _get_offsets(num_grids, num_dim):
        '''
        get offsets for each grid
        input: number of grids and number of dimensions
        output: numpy.ndarray
        '''
        offsets = []
        for i in range(num_grids):
            offset = [int(j) for j in np.binary_repr(i, num_dim)]
            offsets.append(offset)
        return np.array(offsets)

    @staticmethod
    def _get_start_positions(breakpoints):
        '''
        get first positions of unique values in sorted array
        input: numpy.ndarray of breakpoints (non-zero elements of first order
        discrete difference of sorted array)
        output: numpy.ndarray of indices that represent first positions
        '''
        start_pos = breakpoints + 1
        start_pos = np.insert(start_pos, 0, 0)
        return start_pos

    @staticmethod
    def _get_boxes_of_grid(box_id, object_id):
        '''
        get for each box the objects it contains
        input: numpy.ndarray of box ids , numpy.ndarray of object ids
        output: list of lists
        '''
        # Sort box_id and object_id according to box_id
        idx = np.argsort(box_id)
        box_id_sorted = box_id[idx]
        object_id_sorted = list(object_id[idx])
        # Get positions (breakpoints) in vector where box_id_sorted changes
        difference = np.diff(box_id_sorted)
        breakpoints = np.nonzero(difference)[0]
        # Get starting positions based on breakpoints
        start_pos = SparseShiftedComputation._get_start_positions(breakpoints)
        # Get number of objects in each box
        num_obj_per_box = np.diff(np.append(start_pos, len(object_id_sorted)+1))
        # Remove boxes which contain a single object
        idx = np.where(num_obj_per_box == 1)
        start_pos = np.delete(start_pos, idx)
        num_obj_per_box = np.delete(num_obj_per_box, idx)
        # Compute ending positions
        end_pos = start_pos + num_obj_per_box
        boxes = [object_id_sorted[x:y] for x, y in six.moves.zip(start_pos, end_pos)]
        return boxes

    def _get_coordinates(self, data, offset):
        '''
        get for each object the coordinates of the box it falls in
        input: data (numpy.ndarray), and offsets of grid (numpy.ndarray)
        output: numpy.ndarray
        '''
        # Shift data
        data = data + offset[np.newaxis, :]
        # Get coordinates for each object
        coordinates = data / float(self.intervalLength)
        coordinates = coordinates.astype(int)
        pos_idx = np.greater_equal(data, np.ones(data.shape))
        close_idx = np.isclose(data, 1)
        coordinates[np.logical_and(pos_idx, close_idx)] -= 1
        return coordinates

    def _convert_coordinates_to_ids(self, coordinates, num_dim):
        '''
        convert coordinates to a single integer that represents a box
        input: numpy.ndarray of coordinates, number of dimensions
        output: numpy.ndarray of box ids
        '''
        dims = tuple([self.gridResolution for i in range(num_dim)])
        return np.ravel_multi_index(np.transpose(coordinates), dims)

    def _get_pairs_of_grid(self, data, offset, object_id):
        '''
        constructs a grid according to the specified offset and returns
        all pairs of objects that fall within the same grid box
        input: data (np.array), offsets (np.array)
        output: list of tuples (list[(int,int)])
        '''
        coordinates = self._get_coordinates(data, offset)
        # Remove objects that fall out of the grid
        idx = np.where(coordinates > self.gridResolution-1)
        coordinates = np.delete(coordinates, idx[0], 0)
        object_id = np.delete(object_id, idx[0], 0)
        # Convert coordinates to box_ids
        num_dim = len(data[0])
        box_id = self._convert_coordinates_to_ids(coordinates, num_dim)
        # Get objects that fall within each box
        boxes = SparseShiftedComputation._get_boxes_of_grid(box_id, object_id)
        # Construct pairs
        pairs = []
        for box in boxes:
            box.sort()
            pairs += itertools.combinations(box, 2)
        return pairs

    def get_similar_indices(self, data, seed=None, **kwargs):
        '''`get_similar_indices` uses a set of shifted grids to find pairs of
        similar objects in the data

        `get_similar_indices` first projects the (high-dimensional) data set
        onto a low-dimensional space using the `DimReducer`. Then multiple
        shifted grids are created and for each grid the pairs of objects that
        fall within the same grid block are returned.

        Args:
            data (numpy.ndarray): input data with n rows (objects) and p
                                  columns (features)

        Keyword arguments:
            statistics (bool):    if true, dictionary with statistics is added
                                  to output

        Returns:
            dictionary:          contains list of pairs and (if statistics
                                 is true) also a dictionary of statistics
        '''

        if not isinstance(data, np.ndarray):
            raise TypeError('data should be a numpy array')

        if 'normalize' in [key for key in kwargs]:
            normalize = kwargs['normalize']
        else:
            normalize = True

        if 'statistics' in [key for key in kwargs]:
            statistics = kwargs['statistics']
        else:
            statistics = False

        # Reduce dimensionality of data only if a dimReducer is provided
        if self.dimReducer is None:
            reduced_data = data
        else:
            reduced_data = self.dimReducer.fit_transform(data, seed=seed)

        # Get number of dimensions in low-dimensional space
        num_dim = len(reduced_data[0])
        # Get number of grids required
        num_grids = 2**num_dim
        # Compute offset for each grid
        offsets = SparseShiftedComputation._get_offsets(num_grids, num_dim)
        # Normalize reduced data if required
        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            normalized_data = min_max_scaler.fit_transform(reduced_data)
        else:
            normalized_data = reduced_data
        # Compute how much to offset
        offsets = offsets*(self.intervalLength / 2.0)
        # Get object ids
        object_id = np.arange(len(data[:, 0]))
        # Determine pairs for each grid
        pairs = []
        for offset in offsets:
            pairs = pairs + self._get_pairs_of_grid(normalized_data,
                                                    offset, object_id)
        if statistics:
            num_pairs = len(pairs)
            output = {'pairs': set(pairs)}
            statistics_dict = {}
            statistics_dict['num_duplicate_pairs'] = num_pairs-len(output['pairs'])
            output['statistics'] = statistics_dict
        else:
            output = {'pairs': set(pairs)}
        return output

class SparseHybridComputation(SparseShiftedComputation):
    '''
    Combination of SparseComputation and SparseShiftedComputation
    '''

    @staticmethod
    def _get_baseline_boxes(coordinates, box_id, object_id):
        '''
        get for each box of baseline grid the objects it contains and return
        the coordinates that correspond to each box
        input: numpy.ndarray of coordinates, numpy.ndarray of box ids,
               numpy.ndarray of object ids
        output: list that contains boxes (list of lists) and a numpy.ndarray
                that contains the corresponding coordinates
        '''
        # Sort box_id and object_id according to box_id
        idx = np.argsort(box_id)
        box_id_sorted = box_id[idx]
        object_id_sorted = list(object_id[idx])
        coordinates_sorted = coordinates[idx]
        # Get positions (breakpoints) in vector where box_id changes
        difference = np.diff(box_id_sorted)
        breakpoints = np.nonzero(difference)[0]
        # Get unique coordinates based on breakpoints
        unique_coordinates = coordinates_sorted[breakpoints]
        unique_coordinates = np.append(unique_coordinates,
                                       np.array([coordinates_sorted[-1]]),
                                       axis=0)
        # Get starting positions by incrementing breakpoints by 1
        start_pos = SparseHybridComputation._get_start_positions(breakpoints)
        # Get number of objects in each box
        num_obj_per_box = np.diff(np.append(start_pos, len(object_id_sorted)+1))
        # Compute ending positions
        end_pos = start_pos + num_obj_per_box
        boxes = [object_id_sorted[x:y] for x, y in six.moves.zip(start_pos, end_pos)]
        return [boxes, unique_coordinates]

    def _apply_shifted_grid(self, boxes, unique_coordinates, statistics=False):
        '''
        apply shifted grid to unique_coordinates and return pairs of boxes
        that are neighbors. If argument statistics is true, then the function
        returns a dictionary that contains the pairs and a dictionary of
        statistics
        input: list of boxes, numpy.ndarray of corresponding coordinates,
               boolean statistics
        output: list of pairs. Each pair contains indices of objects that
                are similar, dictionary with statistics if statistics is true
        '''
        self.gridResolution = int(self.gridResolution/float(2))
        self.intervalLength = 1/ float(self.gridResolution)
        ssc = SparseShiftedComputation(dimReducer=None,
                                       gridResolution=self.gridResolution)
        normalized_data = unique_coordinates/float(np.amax(unique_coordinates,
                                                     axis=0, keepdims=True))
        output_ssc = ssc.get_similar_indices(normalized_data, normalize=False,
                                              statistics=statistics)

        pairs = []
        for comparison in output_ssc['pairs']:
            pairs += itertools.product(boxes[comparison[0]],
                                       boxes[comparison[1]])
        for box in boxes:
            pairs += itertools.combinations(box, 2)

        output = {'pairs': pairs}
        if statistics:
            output['statistics'] = output_ssc['statistics']
        return output

    def get_similar_indices(self, data, seed=None, **kwargs):
        '''`get_similar_indices` uses a combination of shifted grids and sparse
        computation to find pairs of similar objects in the data

        `get_similar_indices` first projects the (high-dimensional) data set
        onto a low-dimensional space using the `DimReducer`. Then a base grid is
        created to determine some groups of objects. A new data set is created
        where the objects represent the groups of the previous step and the features
        are the coordinates of the groups in the base grid. Then, shifted grid
        is applied to find pairs of neighboring groups. All pairs of objects within
        groups and between neighboring groups are returned.

        Args:
            data (numpy.ndarray): input data with n rows (objects) and p
                                  columns (features)

        Keyword arguments:
            statistics (bool):    if true, dictionary with statistics is added
                                  to output

        Returns:
            dictionary:          contains list of pairs and (if statistics
                                 is true) also a dictionary of statistics
        '''

        if 'statistics' in [key for key in kwargs]:
            statistics = kwargs['statistics']
        else:
            statistics = False

        if self.dimReducer is None:
            reduced_data = data
        else:
            reduced_data = self.dimReducer.fit_transform(data, seed=seed)
        num_dim = len(reduced_data[0])
        # Multiply grid resolution by 2
        self.gridResolution = self.gridResolution*2
        self.intervalLength = 1/ float(self.gridResolution)
        # Get box-coordinates for each object
        object_id = np.arange(len(data[:, 0]))
        coordinates = self._rescale_data(reduced_data)
        # Get objects that fall within each box
        box_id = self._convert_coordinates_to_ids(coordinates, num_dim)
        output = self._get_baseline_boxes(coordinates, box_id, object_id)
        boxes = output[0]
        unique_coordinates = output[1]
        # Apply shifted grid
        return self._apply_shifted_grid(boxes, unique_coordinates,
                                        statistics=statistics)
