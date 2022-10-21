#imports from packages
#from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#import from local modules
from transformers_num import get_transf_num

def construct_preprocessor(opt,features2impute):
    
    numerical_transformer = get_transf_num(opt)

    preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, features2impute),
                    ])

    return(preprocessor)