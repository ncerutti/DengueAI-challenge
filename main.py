"""
Main module of the Dengue project.

Authors:
Emanuele Roppo
ncerutti
OnurKerimoglu
"""

# import packages
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

# import functions from project modules
from rawdata import get_data
from feature_eng import preprocess  # get_features
from processors import construct_preprocessor, construct_FEprocessor
from models import construct_model
from FPE import fit_predict_evaluate
from utilities import get_expname_datetime


def main(options, expname):
    """
    Main function of the Dengue project.

    Args: 
    options: a dictionary containing options for features, preprocessing, model
        Example:
        options = {'features': 'AvgTemp_Prec', 'preprocessing': 'median', 'model': 'LR'}

    Returns: None

    """

    # Load the data
    train_features, test_features = get_data()

    # Preprocess the data
    train_clean, test_clean = preprocess(train_features, test_features)

    # Get features
    # features = get_features(options['features'])

    # Build the preprocessor
    # preprocessor = construct_preprocessor(opt=options['preprocessing']['num'],
    #                                      features2impute=features)

    # Build a feature engineering processor
    # FEprocessor = construct_FEprocessor(opt=options['preprocessing']['FE'],
    #                                         features2eng=features)

    # Build the model
    model,GSparameters = construct_model(opt=options["model"])

    # Build the entire pipeline
    pl = Pipeline(
        steps=[  # ('preprocessor', preprocessor),
            # ('featureeng', FEprocessor), #does not work
            ("model", model)
        ]
    )

    # do the fitting and predictions
    scores = fit_predict_evaluate(
        pl,
        train_clean,
        test_clean,
        test_features,
        GSparameters,
        expname=expname,
        operation = 'test', #'test' 'crossval', 'gridsearch',
        create_submission=True,
    )

    # save options and scores in a pickle
    with open(expname + ".pickle", "wb") as handle:
        pickle.dump((scores, options), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved the scores and options in: {expname}.pickle/png")

    return


if __name__ == "__main__":
    options = {  #'features': 'AvgTemp_Prec_NDVI',
        "features": "AvgTemp_Prec",
        "preprocessing": {"num": "median", "FE": "addlags"},
        "model": "RFR",  #'GBR','RFR', 'DTR'
    }
    # construct an experiment name based on current date time
    expname = get_expname_datetime()
    #expname='test'
    main(options, expname)
