import numpy as np

class Boxes( object ):
    '''This class creates the gridResolution^dimLow boxes and will compute the pairs to compare'''
    def __init__(self, gridResolution):
        if not isinstance(gridResolution,int):
            raise TypeError
        if gridResolution<1:
            raise ValueError
        self.gridResolution = gridResolution

    def get_min (self,data):
        '''
        take the data and return the minimum for each dimension in a list
        input: numpy array
        output: list of length dimLow
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])>3:
            raise ValueError
        n=len(data[0])
        minimum=np.copy(data[0])
        for vec in data:
            for i in range(0,n):
                if vec[i]<minimum[i]:
                    minimum[i]=vec[i]
        return minimum

    def get_max (self, data):
        '''
        take the data and return the max for each dimension in a list
        input: numpy array
        output: list of length dimLow
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])>3:
            raise ValueError
        n=len(data[0])
        maximum=np.copy(data[0])
        for vec in data:
            for i in range(0,n):
                if vec[i]>maximum[i]:
                    maximum[i]=vec[i]
        return maximum

    def rescale_data(self,data):
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])>3:
            raise ValueError
        n=len(data[0])
        maximum=self.get_max(data)
        minimum=self.get_min(data)
        coef=[]
        rescaled_data=[]
        for i in range(0,n):
            try:
                coef.append(float(self.gridResolution)/(float(maximum[i])-float(minimum[i])))
            except ZeroDivisionError:
                coef.append(0)
        for vec in data:
            rescaled_vec=[]
            for i in range(0,n):
                rescaled_vec.append(int((vec[i]-minimum[i])*coef[i]))
            rescaled_data.append(rescaled_vec)
        return np.array(rescaled_data)

    def get_pairs(self,data):
        if not isinstance(data, np.ndarray):
            raise TypeError
        if len(data[0])>3:
            raise ValueError
        n=len(data[0])
        rescaled_data=self.rescale_data(data)
        pairs=[]
        for idx1 in range(0,len(rescaled_data)):
            for idx2 in range(idx1+1,len(rescaled_data)):
                boolean = True
                for i in range(0,n):
                    boolean=boolean and (rescaled_data[idx2][i]<=rescaled_data[idx1][i]+1) and (rescaled_data[idx2][i]>=rescaled_data[idx1][i]-1)
                if boolean:
                    pairs.append((idx1,idx2))
        return pairs
