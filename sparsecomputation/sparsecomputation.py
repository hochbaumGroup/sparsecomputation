import numpy as np
import itertools
from sklearn import preprocessing
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

        coef = float(self.gridResolution) / float(gap)

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

    def get_similar_indices(self, data):
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

        reduced_data = self.dimReducer.fit_transform(data)
        return self._get_pairs(reduced_data)

class SparseShiftedComputation (SparseComputation):

    def _getOffsets(self,nGrid,p):
        '''
        get offsets for each grid
        input: number of grids and number of dimensions
        output: np.array
        '''
        offsets = []
        for i in range(nGrid):
            ls = [int(i) for i in np.binary_repr(i, p)]
            offsets.append(ls)
        return np.array(offsets)

    def _get_boxes(self,BoxID,ObjectID):
        '''
        get for each box the objects that fall within
        input: vector of box ids (np.array), vector of object ids (np.array)
        output: list of lists
        '''
        # Sort BoxID and ObjectID according to BoxID
        idx = np.argsort(BoxID)
        BoxID_sorted = BoxID[idx]
        ObjectID_sorted = list(ObjectID[idx])
        # Get positions (breakpoints) in vector where BoxID changes
        difference = np.diff(BoxID_sorted)
        breakpoints = np.nonzero(difference)[0]
        # Get starting positions by incrementing breakpoints by 1
        starting_positions = breakpoints + 1
        # Add first position in vector as starting position
        starting_positions = np.insert(starting_positions,0,0)
        # Get number of objects in each box
        nObjectsPerBox = np.diff(np.append(starting_positions,len(ObjectID_sorted)+1))
        # Remove boxes which contain a single object
        z = np.where(nObjectsPerBox==1)
        starting_positions = np.delete(starting_positions,z)
        nObjectsPerBox = np.delete(nObjectsPerBox,z)
        # Compute ending positions
        ending_positions = starting_positions + nObjectsPerBox
        numBoxes = len(starting_positions)
        boxes = []
        for i in range(numBoxes):
            boxes.append(ObjectID_sorted[starting_positions[i]:ending_positions[i]])
        return boxes

    def _get_pairs_of_grid(self,data,offset):
        '''`_get_pairs_of_grid` constructs a grid according to the specified
        offset returns all pairs of objects that lie within the same grid block
        input: data (np.array), offsets (np.array)
        output: list of pairs (list[(int,int)])
        '''
        # Get object ids
        ObjectID = np.array(list(range(len(data[:,0]))))
        # Shift data
        data = data + offset[np.newaxis,:]
        # Get interval length
        intervalLength = 1 / float(self.gridResolution)
        # Get coordinates for each object
        coordinates = np.floor(data / float(intervalLength))+1
        coordinates = coordinates.astype(int)
        coordinates[data==1] -= 1
        coordinates -= 1
        # Remove objects that fall out of the grid
        idx = np.where(coordinates>self.gridResolution-1)
        coordinates = np.delete(coordinates,idx[0],0)
        ObjectID = np.delete(ObjectID,idx[0],0)
        # Convert coordinates to BoxIDs
        p = len(data[0])
        dims = tuple([self.gridResolution for i in range(p)])
        BoxID = np.ravel_multi_index(np.transpose(coordinates),dims)
        # Get objects that fall within each box
        boxes = self._get_boxes(BoxID,ObjectID)
        # Construct pairs
        pairs = []
        for box in boxes:
            box.sort()
            pairs += itertools.combinations(box,2)
        return pairs

    def get_similar_indices(self, data):
        '''`get_similar_indices` uses a set of shifted grids to find pairs of
        similar objects in the data

        `get_similar_indices` first projects the (high-dimensional) data set
        onto a low-dimensional space using the `DimReducer`. Then multiple
        shifted grids are created and for each grid the pairs of objects that
        fall within the same grid block are returned.

        Args:
            data (numpy.ndarray): input data with n rows (objects) and p
                                  columns (features)
        Returns:
            (list [(int, int)]): list of pairs. Each pair contains indices of
                                 objects that are similar
        '''
        reduced_data = self.dimReducer.fit_transform(data)
        p = len(reduced_data[0])
        nGrids = 2**p
        offsets = self._getOffsets(nGrids,p)
        # Normalize reduced data
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_data = min_max_scaler.fit_transform(reduced_data)
        # Compute offsets
        intervalLength = 1/ float(self.gridResolution)
        offsets = offsets*(intervalLength/ float(2))
        # Determine pairs
        pairs = []
        for offset in offsets:
            pairs = pairs + self._get_pairs_of_grid(normalized_data,offset)
        return set(pairs)
