import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def basic_cleaning(train_feature_path, train_labels_path, test_feature_path, export=False, out_path=None):
    train_features=pd.read_csv(train_feature_path)
    train_labels=pd.read_csv(train_labels_path)
    train_features=pd.merge(train_features, train_labels, on=['city', 'year','weekofyear'])

    test_features=pd.read_csv(test_feature_path)


    #convert kelvin to celsius
    train_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]=train_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]-273.15
    test_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]=test_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]-273.15

    #dropping rows where no temperature data is available

    train_clean=train_features.dropna(subset=['reanalysis_air_temp_k', 'reanalysis_avg_temp_k','station_avg_temp_c'], how='all')

    # inputation for submission


    test_clean=test_features.copy(deep=True)

    #inputation - temperature data

    train_clean['station_avg_temp_c'].fillna(train_clean['reanalysis_avg_temp_k'], inplace=True)
    train_clean['station_diur_temp_rng_c'].fillna(train_clean['reanalysis_tdtr_k'], inplace=True)
    train_clean['station_max_temp_c'].fillna(train_clean['reanalysis_max_air_temp_k'], inplace=True)
    train_clean['station_min_temp_c'].fillna(train_clean['reanalysis_min_air_temp_k'], inplace=True)

    test_clean['station_avg_temp_c'].fillna(test_clean['reanalysis_avg_temp_k'], inplace=True)
    test_clean['station_diur_temp_rng_c'].fillna(test_clean['reanalysis_tdtr_k'], inplace=True)
    test_clean['station_max_temp_c'].fillna(test_clean['reanalysis_max_air_temp_k'], inplace=True)
    test_clean['station_min_temp_c'].fillna(test_clean['reanalysis_min_air_temp_k'], inplace=True)

    #inputation - vegetation index

    for i in ['ndvi_ne','ndvi_sw','ndvi_nw','ndvi_se']:
        train_clean[i]=train_clean[i].interpolate()
        test_clean[i]=test_clean[i].interpolate()


    #inputation - precipitation level
    train_clean['station_precip_mm'].fillna(train_clean['reanalysis_sat_precip_amt_mm'], inplace=True)
    test_clean['station_precip_mm'].fillna(test_clean['reanalysis_sat_precip_amt_mm'], inplace=True)

    #dropping duplicate columns
    train_clean.drop(['precipitation_amt_mm','reanalysis_sat_precip_amt_mm'],axis=1, inplace=True)
    test_clean.drop(['precipitation_amt_mm','reanalysis_sat_precip_amt_mm'],axis=1, inplace=True)

    #drop useless column
    train_clean.drop('week_start_date', axis=1, inplace=True)
    test_clean.drop('week_start_date', axis=1, inplace=True)

    #encode city as binary variable
    train_clean['city']=train_clean['city'].map({'sj':1, 'iq':0})
    test_clean['city']=test_clean['city'].map({'sj':1, 'iq':0})

    for i in test_clean.columns:
        test_clean[i]=test_clean[i].interpolate()

    if export:
        train_clean.to_csv('train_'+out_path)
        test_clean.to_csv('test_'+out_path)

    return train_clean, test_clean

def export_submission(test_clean, out_path, model):

    test_clean['total_cases']=model.predict(test_clean)
    test_clean['total_cases']=test_clean['total_cases'].astype(int)
    submission=test_clean[['city', 'year', 'weekofyear', 'total_cases']]
    submission['city']=submission['city'].map({1:'sj', 0:'iq'})
    submission.to_csv(out_path, index=False)

    return submission

cleaned=basic_cleaning('../data/dengue_features_train.csv', '../data/dengue_labels_train.csv','../data/dengue_features_test.csv' )

print(cleaned[0].head())

X=cleaned[0].drop('total_cases', axis=1)
Y=cleaned[0]['total_cases']



X_train, X_test, y_train , y_test=train_test_split(X,Y, random_state=42)
dtr=DecisionTreeRegressor(random_state=420)
dtr.fit(X_train, y_train)

rfr=RandomForestRegressor(random_state=420)
rfr.fit(X_train, y_train)




print(mean_absolute_error(y_test, dtr.predict(X_test)))
print(mean_absolute_error(y_test, rfr.predict(X_test)))
