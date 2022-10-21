from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TempLagTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        temp_lag1 = X['station_avg_temp_c'].shift(1)
        temp_lag1.iloc[0] = temp_lag1.iloc[1]
        X["temp_laggged_1"] = temp_lag1
        return X

def get_transf_FE(opt):
    # Preprocessing: feature engineering
    
    if opt=='addlags':
        feature_engineering = TempLagTransformer()
    else:
        raise(ValueError(f'feature enginerring:{opt} option not defined'))
    
    return feature_engineering