from DimReducer import DimReducer

class ApproximatePCA( DimReducer ):

    def __init__( self, dimLow, percRow = 0.01, percCol = 1, minRow = 150, minCol = 150 ):
        self.dimLow = dimLow
        self.percRow = percRow
        self.percCol = percCol
        self.minRow = minRow
        self.minCol = minCol

    def fit_transform( self, data ):
        return data
