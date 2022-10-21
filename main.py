"""
Main module of the Dengue project.

Authors:
Emanuele Roppo
ncerutti
OnurKerimoglu
"""

#import packages
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

#import functions from project modules
from rawdata import get_data
from feature_eng import get_features
from preprocessors import construct_preprocessor
from models import construct_model
from FPE import fit_predict_evaluate
from utilities import get_expname

def main(options, expname):
    """
    Main function of the Dengue project.

    Args: 
    options: a dictionary containing options for features, preprocessing, model
        Example:
        options = {'features': 'AvgTemp_Prec', 'preprocessing': 'median', 'model': 'LR'}

    Returns: None

    """

    #Load the data
    sj_train,iq_train,sj_test_features,iq_test_features = get_data()

    #Get features
    features = get_features(options['features'])

    #Build the preprocessor
    preprocessor = construct_preprocessor(opt=options['preprocessing'],features2impute=features)

    #Build the model
    model = construct_model(opt=options['model'])

    #Build the entire pipeline
    pl = Pipeline(steps=[('preprocessor', preprocessor),
                         ('model', model)])

    # do the fitting and predictions
    scores = fit_predict_evaluate(pl,sj_train,figname=expname) 

    # save options and scores in a pickle
    with open(expname + '.pickle', 'wb') as handle:
        pickle.dump((scores, options), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved the scores and options in: {expname}.pickle/png')
    
    return
    
if __name__ == "__main__":
    options = {#'features': 'AvgTemp_Prec_NDVI',
               'features': 'AvgTemp_Prec', 
               'preprocessing': 'median',
               'model': 'LR'
               }
    #construct an expriment name based on specified options
    expname = get_expname(options)

    main(options, expname)