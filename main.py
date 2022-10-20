"""
Main module of the Dengue project.

Authors:
Emanuele Roppo
ncerutti
OnurKerimoglu
"""

#import packages that will be needed by multiple modules
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import rawdata

def main():
    """
    Main function of the Dengue project.

    Args: none

    Returns: none

    """

    #Load the data
    train_features,train_labels,test_features = rawdata.load()

    #Choose features

    #Imputations
    
    #Encodings

    #Visualise features (?)

    #Define model

    #Split Train-test

    #Predict, Evaluate
    
    #Visualise predictions

    pass
    
if __name__ == "__main__":
    main()