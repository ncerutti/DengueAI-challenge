import pandas as pd
"""
This module contains functions that deal with raw data.
"""

def get_data():
    """
    Read and return the raw data.
    Args: 
    none
    
    Returns:
    train_features,train_labels,test_features: all pandas dataframes
    """
    #define the index columns
    index_cols = [0,1,2]

    #load raw data
    print("Loading raw data")
    features_train = load('data/dengue_features_train.csv',index_cols, 'features_train')
    labels_train = load('data/dengue_labels_train.csv',index_cols, 'labels_train')
    features_test = load('data/dengue_features_test.csv',index_cols, 'features_test')

    return features_train,labels_train,features_test

    pass

def load(path,index_cols,dsetname):

    df = pd.read_csv(path, index_col=index_cols)
    print(f' loaded {dsetname}. Shape:{df.shape}')                        

    return df

    