def get_features(opt):

    #choose features
    if opt == 'AvgTemp_Prec':
        features = ['station_avg_temp_c','precipitation_amt_mm']
    elif opt == 'AvgTemp_Prec_NDVI':
        features = ['station_avg_temp_c','precipitation_amt_mm','ndvi_ne']
    else:
        raise(ValueError(f'features:{opt} option not defined'))

    return features 