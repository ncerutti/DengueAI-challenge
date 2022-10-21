from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def construct_model(opt):
    if opt == 'LR':
        model = LinearRegression()
    elif opt == 'DTR':
        model=DecisionTreeRegressor()
    else:
        raise(ValueError(f'model:{opt} option not defined'))
        
    return model