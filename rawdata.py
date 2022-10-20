import pandas as pd
"""
This module contains functions that deal with raw data.
"""

def load():
    """
    Read and return the raw data.
    Args: 
    none
    
    Returns:
    train_features,train_labels,test_features: all pandas dataframes
    """

    print("Loading raw data...")
    train_features = pd.read_csv('data/dengue_features_train.csv',
                             index_col=[0,1,2])
    print(f' loaded train_features. Shape:{train_features.shape}')                        

    train_labels = pd.read_csv('data/dengue_labels_train.csv',
                           index_col=[0,1,2])
    print(f' loaded train_labels. Shape:{train_labels.shape}')

    test_features = pd.read_csv('data/dengue_features_test.csv',
                             index_col=[0,1,2])
    print(f' loaded test_features. Shape:{test_features.shape}')

    return train_features,train_labels,test_features                  