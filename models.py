from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def construct_model(opt):
    if opt == 'LR':
        model = LinearRegression()
    elif opt == 'DTR':
        model = DecisionTreeRegressor(random_state=420)
    elif opt == 'RFR':
        model = RandomForestRegressor(random_state=420)
    else:
        raise(ValueError(f'model:{opt} option not defined'))
        
    return model