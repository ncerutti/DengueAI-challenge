from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor,XGBRFRegressor


def construct_model(opt):
    if opt == "LR":
        model = LinearRegression()
        GSparameters=None
    elif opt == "DTR":
        model = DecisionTreeRegressor(
            random_state=420
            )
        GSparameters=None
    elif opt == "RFR":
        model = RandomForestRegressor(
            random_state=420, 
            n_estimators=150, #300, 
            min_samples_split=15 #125
        )
        GSparameters = {
                         'model__n_estimators': [100,150,200],
                         'model__min_samples_split': [10, 15, 25],
                        }
    elif opt == "GBR":
        model=GradientBoostingRegressor(
            loss="absolute_error",
            min_samples_split=50,
            n_estimators=20000,
            alpha=0.05)
        GSparameters =  {            
                         #'model__min_samples_split': [50,100,150],
                         'model__n_estimators': [10000,15000,20000],
                         #'model__alpha':[0.03,0.05,0.07],
                        }	    
    elif opt == "XGB":
        model = XGBRegressor(
            n_estimators=10000,
            max_depth=12,
            eta=0.2,
            subsample=0.75,
            colsample_bytree=0.75,
            colsample_bylevel=0.75,
            colsample_bynode=0.75,
            random_state=43
        )
        GSparameters = None
    elif opt == "XGBRF":
        model = XGBRFRegressor(
            n_estimators=10000,
            max_depth=9,
            eta=0.3,
            subsample=0.7,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            colsample_bynode=0.7,
            random_state=420
        )
        GSparameters = None
    else:
        raise (ValueError(f"model:{opt} option not defined"))

    return model,GSparameters
