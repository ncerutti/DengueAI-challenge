#imports from packages
#from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#import from local modules
from transformers_num import get_transf_num
from transformers_FE import get_transf_FE

def construct_preprocessor(opt,features2impute):
    
    numerical_transformer = get_transf_num(opt)

    preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, features2impute)
                    ])

    return(preprocessor)
    
def construct_FEprocessor(opt,features2eng):

    feature_engineering = get_transf_FE(opt)

    preprocessor = ColumnTransformer(
                    transformers=[
                        ('FE', feature_engineering, features2eng)
                    ])

    return(preprocessor)