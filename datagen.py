import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.datasets import make_blobs,make_circles,make_moons
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load(dataset_name:str='blobs', n_samples:int=1000, n_features:int=2, centers:int=3) -> tuple:
    """
    This function returns X,y_true for the given dataset name from [circles, moons, blobs, mnist, sloan, snap].
    The additional parameters besides the dataset name are purely for testing runtime of the blob dataset.

    Paramters:
    ---------
    dataset_name:   string
                    name of the dataset which to load
    n_samples:      int
                    number of samples to generate for blobs
    n_features:     int
                    number of features in blob dataset
    centers:        int
                    number of cluster centers in blob dataset

    Returns:
    -------
    X:  np.ndarray
        dataset
    y_true: np.ndarray
        labels
    """
    loading_functions={
        'circles': make_circles(n_samples=1500,random_state=42,noise=0.1,factor=0.3),
        'moons': make_moons(n_samples=1500,noise=0.1, random_state=42),
        'blobs': make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=180), 
        'mnist': prepare_mnist(),
        'sloan': prepare_sloan(), 
        'snap':prepare_snap()
    }

    return loading_functions[dataset_name]


def prepare_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    #print(X_test.shape, X_train.shape)
    return X_train[:5000],y_train[:5000]



def prepare_sloan():
    #Skyserver_SQL2_27_2018 6_51_39 PM
    #https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey
    df = pd.read_csv('data/sloan.csv')
    
    y_true = df['class'].copy()
    le = LabelEncoder()
    le.fit(y_true)
    y_true = le.transform(y_true)
    
    df =df.drop(columns=['class'])
    X = df.to_numpy()
    scaler = MinMaxScaler(copy=False)
    scaler.fit_transform(X)
    
    return X,y_true

def prepare_snap():
    #https://snap.stanford.edu/data/email-Eu-core.html
    # n_labels = 42
    X =pd.read_csv('data/snap.txt',header=None,delimiter=' ')
    y = pd.read_csv('data/snap_labels.txt',header=None,delimiter=' ')
    y_true = y[1]
    
    n_nodes = 1005
    W = np.zeros((n_nodes,n_nodes))

    for i in X.index:
        u,v = X[0][i], X[1][i]
        W[u,v]=1

    
    return W,y_true