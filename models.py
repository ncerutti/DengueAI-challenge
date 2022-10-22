from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor

def construct_model(opt):
    if opt == 'LR':
        model = LinearRegression(random_state=420)
    elif opt == 'DTR':
        model = DecisionTreeRegressor(random_state=420)
    elif opt == 'RFR':
        model = RandomForestRegressor(random_state=420)
    elif opt == 'XGB':
        model = XGBRFRegressor(n_estimators=10000, max_depth=22, eta=0.1, subsample=0.7, colsample_bytree=0.8,random_state=420)
    else:
        raise(ValueError(f'model:{opt} option not defined'))
        
    return model