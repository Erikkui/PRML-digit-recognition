import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def load_data( data_path ):
    '''
    Loads the data from the specified path.
    
    Inputs:
    data_path: path to the directory containing the data files, each file containing measurement of a digit
    in a N x 3 matrix. 
    
    Returns:
    data: a list of numpy arrays, each array containing the measurement of a digit.
    labels: a list of labels corresponding to the data.
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
                    
    return data, np.array(labels)

def preprocess_data( data, N_interp = 128 ):
    '''
    Preprocesses the data by interpolating to equal lengths and standardizing.
    
    Inputs:
    Data: list of numpy arrays, each array containing the measurement of a digit.
    N_interp: number of points to interpolate the variable-length data to.
    
    Returns:
    data: a list of numpy arrays, each array containing the measurement of a digit after preprocessing.
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
        
def kkn_classifier( X_train, y_train, sample, k = 3 ):
    '''
    Returns the most frequent digit label for a given sample basedf on the k-nearest neighbors.
    
    Inputs:
    X_train: array of training samples.
    y_train: array of training labels.
    sample: sample to classify.
    k: number of neighbors to consider.
    
    Returns:
    label: the most frequent label among the k-nearest neighbors.
    '''
    
    # Make sure sample is equal length to training samples
    N_train = X_train[0].shape[0]
    if sample.shape[0] != N_train:
        sample = preprocess_data( [sample], N_train )[0]
    
    # Compute distances to all training samples
    distances = np.zeros( len( X_train ) )
    for ii, train_sample in enumerate( X_train ):
        distances[ii] = np.linalg.norm( train_sample - sample, ord = "fro" )    # Frobenius norm
     
    # Find k-nearest neighbors
    sorted_inds = np.argsort( distances )   # Sort indices based on distances
    nearest_labes = y_train[ sorted_inds[:k] ]  # Get labels of k-nearest neighbors
    vals, counts = np.unique( nearest_labes, return_counts = True )     # Count the number of occurrences of each label
    label = vals[ np.argmax( counts ) ]     # Get the most frequent label
    
    return label

def results( y_test, y_classified ):
    '''
    Computes the accuracy and confusion matrix of the classification.
    
    Inputs: 
    y_test: true labels of the test samples.
    y_classified: labels classified by the classifier.
    
    Returns:
    '''
    
    accuracy = np.sum( y_test == y_classified ) / len( y_test )
    print( 'Accuracy: ', accuracy )
    
    cm = confusion_matrix( y_test, y_classified )
    sns.heatmap( cm, annot = True )
    plt.title('Accuracy: ' + str(accuracy))
    plt.show()


def main():
    
    data_path = os.getcwd() + '/data/digits_3d/training_data'
    data, labels = load_data( data_path )
    data = preprocess_data( data )    
    
    X_train, X_test, y_train, y_test = train_test_split( data, labels, test_size=0.25, stratify = labels ) 
    # print( y_test)
    # Classify test samples
    y_classified = []
    for ii, test_sample in enumerate( X_test ):
        label = kkn_classifier( X_train, y_train, test_sample )
        y_classified.append( label )
        
    results( y_test, y_classified )
    
main()
        
