import pytest
import numpy as np


@pytest.fixture
def SC():
    """Simple Sparse Computation object"""
    from sparsecomputation import SparseComputation

    return SparseComputation(None, resolution=4,
                             method='block_shifting',
                             rescale=None)


@pytest.fixture
def data():
    """Simple data object"""
    return np.array([
        [0, 0],
        [0.375, 0.375],
        [0.625, 0.375],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.5, 0.5],
        [1.0, 1.0],
        ])


@pytest.fixture
def IDs():
    """Grid indices of points in data"""
    return np.array([
        [0, 0],
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [2, 2],
        [4, 4],
    ])


@pytest.fixture
def boxDict():
    """Dict of points in each block"""
    return {
        (0, 0): [0, ],
        (1, 1): [1, ],
        (2, 1): [2, ],
        (1, 2): [3, ],
        (2, 2): [4, 5, ],
        (4, 4): [6, ],
    }


@pytest.fixture
def reps():
    """Numpy array of representative objects"""
    return np.array([
        [0.125, 0.125],
        [0.375, 0.375],
        [0.375, 0.625],
        [0.625, 0.375],
        [0.625, 0.625],
        [1.125, 1.125],
        ])


@pytest.fixture
def pairs():
    """List of pairs"""
    return [
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        ]


def test_init(SC):
    assert SC.method == 'block_shifting'
    assert SC.rescale is None
    assert SC.resolution == 4.0
    assert SC.distance == 0.25
    assert SC.dimReducer is None
    assert SC.stats is None


def test_resolution_setter(SC):
    SC.resolution = 5
    assert SC.resolution == 5.0
    assert SC.distance == 0.2


def test_init_both_none_distance_resolution():
    from sparsecomputation import SparseComputation

    with pytest.raises(ValueError):
        SparseComputation(None, resolution=4, distance=1.0)

    with pytest.raises(ValueError):
        SparseComputation(None)


def test_project_onto_grid(SC, data, IDs):
    np.testing.assert_equal(SC._project_onto_grid(data, 0.25), IDs)


def test_get_box_id(SC, IDs, boxDict):
    assert SC._get_box_dict(IDs) == boxDict


def test_generate_shifts(SC):
    assert SC._generate_shifts(2) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_pairs_within_block(SC, boxDict):
    assert SC._select_within_block_pairs(boxDict) == [
        (4, 5),
    ]


def test_create_representatives(SC, boxDict, reps):
    boxes = sorted(boxDict.keys())
    np.testing.assert_allclose(SC._create_representatives(boxes), reps)


def test_block_enumeration(SC, data, pairs):
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC._block_enumeration(data)
        ])
    assert sortedPairs == pairs


def test_object_shifting(SC, data, pairs):
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC._object_shifting(data)
        ])
    assert sortedPairs == pairs


def test_block_shifting(SC, data, pairs):
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC._block_shifting(data)
        ])
    assert sortedPairs == pairs


@pytest.mark.parametrize("data,expected", [
    (data(), data()),
    (np.array([
        [1.0, 0.0],
        [1.5, 1.5],
        [2.0, 3.0],
        ]),
     np.array([
         [0.0, 0.0],
         [0.5, 0.5],
         [1.0, 1.0],
         ])
     ),
])
def test_rescale_min_max(SC, data, expected):
    np.testing.assert_allclose(SC._rescale_min_max(data), expected)


def test_rescale_data(SC, data):
    SC.rescale = None
    np.testing.assert_allclose(SC._rescale_data(data), data)

    SC.rescale = 'min_max'
    np.testing.assert_allclose(SC._rescale_data(data), data)

    with pytest.raises(ValueError):
        SC.rescale = 'test'
        SC._rescale_data(data)


def test_select_pairs_no_array(SC):
    with pytest.raises(TypeError):
        SC.select_pairs('test')


def test_select_pairs(SC, data, pairs):
    SC.method = 'block_enumeration'
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])
    assert sortedPairs == pairs

    SC.method = 'block_shifting'
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])
    assert sortedPairs == pairs

    SC.method = 'object_shifting'
    sortedPairs = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])
    assert sortedPairs == pairs

    SC.method = 'test'
    with pytest.raises(ValueError):
        SC.select_pairs(data)


def test_fit_transform_dim_reducer(SC):
    from tests.test_dimreducer import PCA, data

    data = data()
    pca = PCA()

    SC.dimReducer = pca

    assert SC.select_pairs(data) == list()


def test_select_pairs_real_data(SC):
    from sparsecomputation import PCA
    from sklearn.datasets import load_iris

    data, _ = load_iris(return_X_y=True)

    pca = PCA(3)

    SC.dimReducer = pca
    SC.resolution = 25
    SC.rescale = 'min_max'

    SC.method = 'block_enumeration'
    sortedPairsBE = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])

    SC.method = 'block_shifting'
    sortedPairsBS = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])
    assert sortedPairsBE == sortedPairsBS

    SC.method = 'object_shifting'
    sortedPairsOS = sorted([
        tuple(sorted(x)) for x in SC.select_pairs(data)
        ])
    assert sortedPairsBE == sortedPairsOS
