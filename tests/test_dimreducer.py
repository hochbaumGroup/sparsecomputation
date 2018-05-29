import numpy as np
import pytest
from mock import patch


@pytest.fixture
def DR():
    from sparsecomputation.dimreducer import DimReducer

    return DimReducer(2)


@pytest.fixture
def PCA():
    from sparsecomputation import PCA

    return PCA(2)


@pytest.fixture
def APCA():
    from sparsecomputation import ApproximatePCA

    return ApproximatePCA(2, fracRow=0.01, fracCol=0.01, minRow=1, minCol=1)


@pytest.fixture
def APCA():
    from sparsecomputation import ApproximatePCA

    return ApproximatePCA(2, fracRow=0.01, fracCol=0.01, minRow=1, minCol=1)


@pytest.fixture
def data():
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        ])


@pytest.fixture
def pcaResult():
    return np.array([
        [-1, - 1 / 3.0],
        [0, 2 / 3.0],
        [1, - 1 / 3.0],
    ])


@pytest.mark.parametrize("dimReducer", [DR(), PCA(), APCA()])
def test_dim_reducer_init(dimReducer):
    '''Check if dimension is assigned correctly'''
    assert dimReducer.dimLow == 2


def test_pca_raise_init():
    '''Test if PCA raises errors on incorrect dimension'''
    from sparsecomputation import PCA

    with pytest.raises(TypeError):
        PCA('a')

    with pytest.raises(ValueError):
        PCA(-1)


def test_input_exceptions_fit_transform(PCA):
    data = [[1, ]]

    with pytest.raises(TypeError):
        PCA.fit_transform(data)

    data = np.array(data)
    with pytest.raises(ValueError):
        PCA.fit_transform(data)


def test_pca(PCA, data, pcaResult):
    np.testing.assert_allclose(PCA.fit_transform(data), pcaResult, atol=1e-8)


def test_fit_transform_separate_pca(PCA, data, pcaResult):
    PCA.fit(data)
    np.testing.assert_allclose(PCA.transform(data), pcaResult,
                               atol=1e-8)


def test_dimension_consistance_apca(APCA):
    data = np.random.normal(0, 1, size=(400, 4))
    APCA.fracRow = 0.5
    out = APCA.fit_transform(data)

    assert out.shape == (400, 2)


def test_approx_pca_init(APCA):
    assert APCA.fracRow == 0.01
    assert APCA.fracCol == 0.01
    assert APCA.minRow == 1
    assert APCA.minCol == 1


def test_column_probability_apca(APCA, data):
    expectedResult = np.array([2 / 3.0, 1 / 3.0, 0.0])

    np.testing.assert_allclose(APCA._get_proba_col(data), expectedResult)


def test_row_probability_apca(APCA, data):
    expectedResult = np.array([1 / 3.0, 1 / 3.0, 1 / 3.0])

    np.testing.assert_allclose(APCA._get_proba_row(data), expectedResult)


@patch('numpy.random.choice')
def test_col_reduction_apca(m, APCA, data):
    expectedResult = np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
    ]) / np.sqrt(np.array([2 / 3.0, 1 / 3.0]) * 2)

    m.return_value = [0, 1]

    np.testing.assert_allclose(APCA._col_reduction(data), expectedResult,
                               atol=1e-8)


@patch('numpy.random.choice')
def test_row_reduction_apca(m, APCA, data):
    expectedResult = np.array([
        [1, 0, 0],
    ])

    m.return_value = [0, ]

    np.testing.assert_allclose(APCA._row_reduction(data), expectedResult,
                               atol=1e-8)


@pytest.mark.parametrize("dimReducer", [PCA(), APCA()])
def test_dimreducer_fit_transform_not_array(dimReducer):
    with pytest.raises(TypeError):
        dimReducer.fit_transform('test')


@patch('numpy.random.choice')
def test_fit_transform_apca(m, APCA, data, pcaResult):
    APCA.minRow = 3
    APCA.minCol = 3

    m.return_value = (0, 1, 2)

    np.testing.assert_allclose(APCA.fit_transform(data), pcaResult,
                               atol=1e-8)


@patch('numpy.random.choice')
def test_fit_transform_separate_apca(m, APCA, data, pcaResult):
    APCA.minRow = 3
    APCA.minCol = 3
    APCA.fracCol = 1.0

    m.return_value = (0, 1, 2)

    APCA.fit(data)
    np.testing.assert_allclose(APCA.transform(data), pcaResult,
                               atol=1e-8)
