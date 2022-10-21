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
    sj_train,iq_train: 
        pandas data frames containing features and labels (total_cases) to be used for training 
    sj_test_features,iq_test_features: 
        pandas dataframes containing only features, to be used for predicting
    """
    #define the index columns
    index_cols = [0,1,2]

    #load raw data
    print("Loading raw data")
    
    train_features=pd.read_csv('data/dengue_features_train.csv')
    train_labels=pd.read_csv('data/dengue_labels_train.csv')
    train_features=pd.merge(train_features, train_labels, on=['city', 'year','weekofyear'])

    test_features=pd.read_csv('data/dengue_features_test.csv')

    return train_features,test_features
    
    #split the data into cities
    #sj_train, iq_train = split_cities(features_train.join(labels_train))
    #sj_test_features, iq_test_features = split_cities(features_test)

    #return sj_train,iq_train,sj_test_features,iq_test_features
    
def load(path,index_cols,dsetname):

    df = pd.read_csv(path, index_col=index_cols)
    print(f' loaded {dsetname}. Shape:{df.shape}')                        

    return df

def split_cities(df):

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

    