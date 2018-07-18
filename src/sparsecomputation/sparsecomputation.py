from itertools import product, combinations
import six.moves
import numpy as np


class SparseComputation(object):
    def __init__(
        self,
        dim_reducer,
        distance=None,
        resolution=None,
        method="block_shifting",
        rescale="min_max",
    ):
        self.dimReducer = dim_reducer

        if (distance is not None) and (resolution is not None):
            raise ValueError(
                "Please set either the distance parameter or the"
                + "resolution parameter but not both."
            )
        elif resolution is not None:
            self.distance = 1 / float(resolution)
        elif distance is not None:
            self.distance = distance
        else:
            raise ValueError(
                "Either the parameter resolution or distance" + "should be set"
            )

        self.rescale = rescale
        self.method = method
        self.stats = None

    @property
    def resolution(self):
        return 1 / float(self.distance)

    @resolution.setter
    def resolution(self, x):
        self.distance = 1 / float(x)

    def _rescale_min_max(self, data, eps=1e-8):
        """Rescale the data to interval [0, 1) in each dimension.
        Args:
            data (n x p numpy array): Data to rescale
            eps=1e-8 (float): Largest data point is projected to 1-eps
        Returns:
            rescaledData (n x p numpy array): Rescaled data
        """
        maximum = np.amax(data, axis=0, keepdims=True)
        minimum = np.amin(data, axis=0, keepdims=True)

        gap = maximum - minimum
        gap = np.where(gap > 0, gap, 1.0)

        rescaledData = (data - minimum) / gap

        rescaledData = np.where(rescaledData >= 1.0, 1.0 - eps, rescaledData)
        return rescaledData

    def _rescale_data(self, data):
        if self.rescale is None:
            return data
        elif self.rescale == "min_max":
            return self._rescale_min_max(data)
        else:
            raise ValueError(
                "Current rescaling method: %s is not defined." % self.rescale
                + 'Set self.rescale to "min_max" or None.'
            )

    def _project_onto_grid(self, data, distance):
        """Project onto a grid with block width `distance`.
        Divide each datapoint along each dimension by the block width and round
        down.
        Args:
            data (n x p numpy array): data to project
            distance (int): grid width
        Returns:
            Grid indices (n x p numpy array)
        """
        projectedData = data / float(distance)
        projectedData = np.floor(projectedData).astype("int")
        return projectedData

    def _get_box_dict(self, data):
        """Identify groups of row entries with same vector in array.
        Args:
            data (n x p numpy array): row vectors to group by array.
        Returns:
            Dict of unique entries with list of corresponding entries in array.
        """
        boxDict = {}
        for i, obs in enumerate(data):
            boxID = tuple(obs)
            if boxID not in boxDict:
                boxDict[boxID] = []
            boxDict[boxID].append(i)
        return boxDict

    def _create_representatives(self, boxIDs):
        """Generate representatives at the center of each nonempty blocks
        Args:
            boxIds (list): List of n' nonempty box ids (tuples)
        Returns:
            n' x p numpy array with location of the representatives
        """
        repData = np.array(boxIDs, dtype=np.float64)
        repData += 0.5
        repData = repData * self.distance
        return repData

    def _generate_shifts(self, numDims):
        """Generate unit direction vector for data shifts
        Args:
            numDims (int): Number of dimensions
        Returns:
            List of all binary vectors of width numDims except all ones.
        """
        shifts = []
        for i in range(2 ** numDims):
            shifts.append(tuple([int(x) for x in np.binary_repr(i, numDims)]))
        return shifts

    def _block_enumeration(self, data):
        """Identify pairs by enumerating adjacent blocks
        Args:
            data (n x p numpy array): vectors corresponding to the observations
        Returns:
            List of tuples where each tuple is a pair.
        """
        numDims = data.shape[1]

        rescaledData = self._rescale_data(data)
        boxIDs = self._project_onto_grid(rescaledData, self.distance)
        boxDict = self._get_box_dict(boxIDs)

        pairs = []

        increments = tuple(
            increment
            for increment in product(range(-1, 2), repeat=numDims)
            if increment > ((0,) * numDims)
        )

        # initialize stats
        numAdjacentBoxes = 0
        numNonemptyAdjacentBoxes = 0

        for boxID, objects in boxDict.items():
            pairs += combinations(objects, 2)
            for increment in increments:
                incrementedID = tuple(
                    a + b for a, b in six.moves.zip(boxID, increment)
                )

                numAdjacentBoxes += 1

                if incrementedID in boxDict:
                    pairs += product(objects, boxDict[incrementedID])

                    numNonemptyAdjacentBoxes += 1

        # assign stats
        stats = {}
        stats["numBoxes"] = len(boxDict)
        stats["numUniquePairs"] = len(pairs)
        stats["numAdjacentBoxes"] = numAdjacentBoxes
        stats["numNonemptyAdjacentBoxes"] = numNonemptyAdjacentBoxes
        stats["numEmptyAdjacentBoxes"] = (
            numAdjacentBoxes - numNonemptyAdjacentBoxes
        )
        self.stats = stats

        return pairs

    def _select_within_block_pairs(self, boxDict):
        """Select all possible pairs within each box.
        Args:
            boxDict (dict): objects per box
        Returns:
            List of tuple where each tuple is a pair.
        """
        pairs = []
        for objects in boxDict.values():
            pairs += combinations(objects, 2)

        # rename pairs such that a < b for (a, b)
        pairs = [(a, b) if a < b else (b, a) for a, b in pairs]

        return pairs

    def _object_shifting(self, data):
        """Identify pairs by shifting objects
        Args:
            data (n x p numpy array): vectors corresponding to the observations
        Returns:
            List of tuples where each tuple is a pair.
        """
        rescaledData = self._rescale_data(data)
        shifts = self._generate_shifts(data.shape[1])

        pairs = set()
        numPairs = 0

        for shift in shifts:
            shiftedData = rescaledData + [
                float(x) * self.distance for x in shift
            ]
            boxIDs = self._project_onto_grid(shiftedData, 2 * self.distance)
            boxDict = self._get_box_dict(boxIDs)
            shiftPairs = self._select_within_block_pairs(boxDict)

            numPairs += len(shiftPairs)
            pairs = pairs.union(shiftPairs)

        stats = {}
        stats["numUniquePairs"] = len(pairs)
        stats["numTotalPairs"] = numPairs
        stats["numDuplicatePairs"] = numPairs - len(pairs)
        stats["numShifts"] = len(shifts)
        self.stats = stats

        return list(pairs)

    def _block_shifting(self, data):
        """Identify pairs by computing non-empty block representatives and
        applying object shifting to the representatives.
        Args:
            data (n x p numpy array): vectors corresponding to the observations
        Returns:
            List of tuples where each tuple is a pair.
        """
        rescaledData = self._rescale_data(data)
        boxIDs = self._project_onto_grid(rescaledData, self.distance)
        boxDict = self._get_box_dict(boxIDs)
        boxes = list(boxDict.keys())

        repData = self._create_representatives(boxes)

        scObject = SparseComputation(
            None,
            distance=self.distance,
            method="object_shifting",
            rescale=None,
        )
        adjacentBoxes = scObject.select_pairs(repData)

        pairs = self._select_within_block_pairs(boxDict)
        numWithinBlockPairs = len(pairs)

        for box1, box2 in adjacentBoxes:
            pairs += product(boxDict[boxes[box1]], boxDict[boxes[box2]])

        stats = scObject.stats
        stats["numUniquePairs"] = len(pairs)
        stats["numTotalPairs"] += numWithinBlockPairs
        self.stats = stats

        return pairs

    def select_pairs(self, data, seed=None):
        """Applies dimension reduction and selects pairs that are close in the
        low-dimensional space.
        Args:
            data (n x p numpy array): vectors corresponding to the observations
        Returns:
            List of tuples where each tuple is a pair.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data should be a numpy array")

        # Reduce dimensionality of data only if a dimReducer is provided
        if self.dimReducer is None:
            reducedData = data
        else:
            reducedData = self.dimReducer.fit_transform(data, seed=seed)

        if self.method == "block_enumeration":
            return self._block_enumeration(reducedData)
        elif self.method == "object_shifting":
            return self._object_shifting(reducedData)
        elif self.method == "block_shifting":
            return self._block_shifting(reducedData)
        else:
            raise ValueError(
                "Current method: %s is not defined. " % self.method
                + "Set self.method to"
                + "'block_enumeration', 'object_shifting', or "
                + "'block_shifting' (default)."
            )
