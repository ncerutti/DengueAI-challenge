from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def construct_preprocessor(opt,features2impute):
    
    # Preprocessing for numerical data
    if opt=='median':
        numerical_transformer = SimpleImputer(strategy='median')
    else:
        raise(ValueError(f'preprocessor:{opt} option not defined'))

    # # Preprocessing for categorical data, ordinal encoder
    # categorical_ord_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('label', OrdinalEncoder())
    # ])

    # # Preprocessing for categorical data, onehot encoder
    # categorical_onehot_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('label', OneHotEncoder())
    # ])

    preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, features2impute),
                        #('cat_ord', categorical_ord_transformer, ordinal_encode_cols),
                        #('cat_onehot', categorical_onehot_transformer, one_hot_encode_cols)
                    ])

    return(preprocessor)

    