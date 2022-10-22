from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from prophet import Prophet


def construct_model(opt):
    if opt == "LR":
        model = LinearRegression()
    elif opt == "DTR":
        model = DecisionTreeRegressor(random_state=420)
    elif opt == "RFR":
        model = RandomForestRegressor(
            random_state=420, n_estimators=300, min_samples_split=125
        )
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
    elif opt == "GBR":
        model = GradientBoostingRegressor(
            loss="absolute_error",
            min_samples_split=50,
            n_estimators=15000,
            learning_rate=0.05
        )
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
    elif opt == "PRO":
        model = Prophet()    
    else:
        raise (ValueError(f"model:{opt} option not defined"))

    return model
