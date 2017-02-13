import numpy as np

class Boxes( object ):
    '''This class creates the gridResolution^dimLow boxes and will compute the pairs to compare'''
    def __init__(self, gridResolution, data):
        if not isinstance(gridResolution,int):
            raise TypeError
        if gridResolution<1:
            raise ValueError
        self.gridResolution = gridResolution
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])>3:
            raise ValueError
        self.data=data
        self.len=len(data[0])

    def get_min (self):
        '''
        take the data and return the minimum for each dimension in a list
        output: list of length dimLow
        '''
        minimum=self.data[0]
        for vec in self.data:
            for i in range(0,self.len):
                if vec[i]<minimum[i]:
                    minimum[i]=vec[i]
        print minimum
        return minimum

    def get_max (self):
        '''
        take the data and return the max for each dimension in a list
        output: list of length dimLow
        '''
        maximum=self.data[0]
        for vec in self.data:
            for i in range(0,self.len):
                if vec[i]>maximum[i]:
                    maximum[i]=vec[i]
        print maximum
        return maximum

    def rescale_data(self):
        maximum=self.get_max()
        minimum=self.get_min()
        coef=[]
        rescaled_data=[]
        for i in range(0,self.len):
            try:
                print maximum[i]
                print minimum[i]
                coef.append(float(self.gridResolution)/(float(maximum[i])-float(minimum[i])))
            except ZeroDivisionError:
                coef.append(0)
        for vec in self.data:
            rescaled_vec=[]
            for i in range(0,self.len):
                rescaled_vec.append(int((vec[i]-minimum[i])/coef[i]))
            rescaled_data.append(rescaled_vec)
        return rescaled_data

    def get_pairs(self):
        rescaled_data=self.rescale_data()
        pairs=[]
        for idx1 in range(0,len(rescaled_data)):
            for idx2 in range(idx1+1,len(rescaled_data)):
                boolean = TRUE
                for i in range(0,self.len):
                    boolean=boolean and (rescaled_data[idx2][i]<=rescaled_data[idx1][i]+1) and (rescaled_data[idx2][i]>=rescaled_data[idx1][i]-1)
                if boolean:
                    pairs.append((idx1,idx2))
        return pairs
