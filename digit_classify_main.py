import os

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline

def load_data( data_path ):
    '''
    Data path: path to the directory containing the data files, each file containing measurement of a digit
    in a N x 3 matrix. 
    
    Returns a list of numpy arrays, each array containing the measurement of a digit.
    '''
    files = os.scandir( data_path )

    data = []
    labels = []
    for file in files:
        filename = file.name
        if filename.endswith('.csv'):
            number_data = np.loadtxt(data_path + '/' + filename, delimiter=',')
            label = int( filename.split('_')[1] )
            
            data.append( number_data )
            labels.append( label )      
                    
    return data, labels

def preprocess_data( data, N_interp = 128 ):
    '''
    Data: list of numpy arrays, each array containing the measurement of a digit.
    N_interp: number of points to interpolate the variable-length data to.
    
    Returns a list of numpy arrays, each array containing the measurement of a digit after preprocessing.
    Preprocessing contains interpolation to equal elngths, and standardization.
    '''
    scaler = StandardScaler()
    
    for ii, sample in enumerate( data ):
        N_sample, dims = sample.shape
        
        tt = np.linspace(0, 1, N_sample)
        tt_interp = np.linspace(0, 1, N_interp)
        
        # First, interpolation (for each dimension separately)
        sample_interp = np.zeros( (N_interp, dims) )
        for dim in range( dims ):          
            dim_interp = np.interp( tt_interp, tt, sample[:, dim] )
            sample_interp[:, dim] = dim_interp
            
        # Then standardization
        data[ii] = scaler.fit_transform( sample_interp )
        
    return data
        


    

def main():
    
    data_path = os.getcwd() + '/data/digits_3d/training_data'
    data, labels = load_data( data_path )
    data = preprocess_data( data )    
    X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.2, stratify = labels ) 

    


main()
        
