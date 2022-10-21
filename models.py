from sklearn.linear_model import LinearRegression

def construct_model(opt):
    if opt == 'LR':
        model = LinearRegression()
    else:
        raise(ValueError(f'model:{opt} option not defined'))
        
    return model