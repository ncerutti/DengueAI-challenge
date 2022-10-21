from sklearn.impute import SimpleImputer

def get_transf_num(opt):
    # Preprocessing for numerical data
    
    if opt=='median':
        numerical_transformer = SimpleImputer(strategy='median')
    else:
        raise(ValueError(f'numerical transformer:{opt} option not defined'))
    
    return numerical_transformer
