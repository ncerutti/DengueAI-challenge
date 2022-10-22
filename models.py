from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import GradientBoostingRegressor


def construct_model(opt):
    if opt == "LR":
        model = LinearRegression()
        GSparameters=None
    elif opt == "DTR":
        model = DecisionTreeRegressor(random_state=420)
        GSparameters=None
    elif opt == "RFR":
        model = RandomForestRegressor(
            random_state=420, n_estimators=300, min_samples_split=125
        )
        GSparameters = {
                         'model__n_estimators': [100,150,200],
                         'model__min_samples_split': [10, 15, 25],
                        }
    elif opt == "XGB":
        model = XGBRFRegressor(
            n_estimators=1000,
            max_depth=22,
            eta=0.1,
            subsample=0.7,
            colsample_bytree=0.8,
            random_state=420,
        )
        GSparameters =  {
                         'model__n_estimators': [500,1000,1500],
                         'model__max_depth': [10, 15, 20],
                        }
    elif opt == "GBR":
        model=GradientBoostingRegressor(loss="absolute_error", min_samples_split=100, n_estimators=1000, alpha=0.05)
        GSparameters =  {            
                         #'model__min_samples_split': [50,100,150],
                         'model__n_estimators': [2000,3000,4000],
                         'model__alpha':[0.03,0.05,0.07],
                        }
    else:
        raise (ValueError(f"model:{opt} option not defined"))

    return model,GSparameters
