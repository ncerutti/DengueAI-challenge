def preprocess(train_features,test_features):
    #convert kelvin to celsius
    train_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]=train_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]-273.15
    test_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]=test_features[['reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k']]-273.15

    #dropping rows where no temperature data is available

    train_clean=train_features.dropna(subset=['reanalysis_air_temp_k', 'reanalysis_avg_temp_k','station_avg_temp_c'], how='all')

    # inputation for submission
    na_col= ['reanalysis_air_temp_k', 'reanalysis_avg_temp_k','station_avg_temp_c']

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

    return train_clean, test_clean

def get_features(opt):

    #choose features
    if opt == 'AvgTemp_Prec':
        features = ['station_avg_temp_c','precipitation_amt_mm']
    elif opt == 'AvgTemp_Prec_NDVI':
        features = ['station_avg_temp_c','precipitation_amt_mm','ndvi_ne']
    else:
        raise(ValueError(f'features:{opt} option not defined'))

    return features