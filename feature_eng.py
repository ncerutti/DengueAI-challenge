import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def preprocess(train_features, test_features):

    cleaned = basic_cleaning(train_features, test_features)
    train_data = cleaned[0]
    test_data = cleaned[1].copy()

    train_data = add_population_data(train_data)
    train_data = add_temp_dew(train_data)
    train_data = add_lag_cols(train_data)

    test_data = add_population_data(test_data)
    test_data = add_temp_dew(test_data)
    test_data = add_lag_cols(test_data)

    train_data.drop(
        [
            "reanalysis_tdtr_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_air_temp_k",
            "year",
        ],
        axis=1,
        inplace=True,
    )

    test_data.drop(
        [
            "reanalysis_tdtr_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_air_temp_k",
            "year",
        ],
        axis=1,
        inplace=True,
    )

    train_data.to_csv("train_processed.csv", index=False)
    test_data.to_csv("test_processed.csv", index=False)
    # Y=train_data['total_cases']
    # X=train_data.drop('total_cases', axis=1)
    return train_data, test_data


def basic_cleaning(
    train_features,
    test_features,
    export=False,
    out_path=None,
):
    # convert kelvin to celsius
    train_features[
        [
            "reanalysis_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_dew_point_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_tdtr_k",
        ]
    ] = (
        train_features[
            [
                "reanalysis_air_temp_k",
                "reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k",
                "reanalysis_max_air_temp_k",
                "reanalysis_min_air_temp_k",
                "reanalysis_tdtr_k",
            ]
        ]
        - 273.15
    )
    test_features[
        [
            "reanalysis_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_dew_point_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_tdtr_k",
        ]
    ] = (
        test_features[
            [
                "reanalysis_air_temp_k",
                "reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k",
                "reanalysis_max_air_temp_k",
                "reanalysis_min_air_temp_k",
                "reanalysis_tdtr_k",
            ]
        ]
        - 273.15
    )

    # dropping rows where no temperature data is available

    train_clean = train_features.dropna(
        subset=["reanalysis_air_temp_k", "reanalysis_avg_temp_k", "station_avg_temp_c"],
        how="all",
    )

    # inputation for submission

    test_clean = test_features.copy(deep=True)

    # inputation - temperature data

    train_clean["station_avg_temp_c"].fillna(
        train_clean["reanalysis_avg_temp_k"], inplace=True
    )
    train_clean["station_diur_temp_rng_c"].fillna(
        train_clean["reanalysis_tdtr_k"], inplace=True
    )
    train_clean["station_max_temp_c"].fillna(
        train_clean["reanalysis_max_air_temp_k"], inplace=True
    )
    train_clean["station_min_temp_c"].fillna(
        train_clean["reanalysis_min_air_temp_k"], inplace=True
    )

    test_clean["station_avg_temp_c"].fillna(
        test_clean["reanalysis_avg_temp_k"], inplace=True
    )
    test_clean["station_diur_temp_rng_c"].fillna(
        test_clean["reanalysis_tdtr_k"], inplace=True
    )
    test_clean["station_max_temp_c"].fillna(
        test_clean["reanalysis_max_air_temp_k"], inplace=True
    )
    test_clean["station_min_temp_c"].fillna(
        test_clean["reanalysis_min_air_temp_k"], inplace=True
    )

    # inputation - vegetation index

    for i in ["ndvi_ne", "ndvi_sw", "ndvi_nw", "ndvi_se"]:
        train_clean[i] = train_clean[i].interpolate()
        test_clean[i] = test_clean[i].interpolate()

    # inputation - precipitation level
    train_clean["station_precip_mm"].fillna(
        train_clean["reanalysis_sat_precip_amt_mm"], inplace=True
    )
    test_clean["station_precip_mm"].fillna(
        test_clean["reanalysis_sat_precip_amt_mm"], inplace=True
    )

    # dropping duplicate columns
    train_clean.drop(
        ["precipitation_amt_mm", "reanalysis_sat_precip_amt_mm"], axis=1, inplace=True
    )
    test_clean.drop(
        ["precipitation_amt_mm", "reanalysis_sat_precip_amt_mm"], axis=1, inplace=True
    )

    # drop useless column
    train_clean.drop("week_start_date", axis=1, inplace=True)
    test_clean.drop("week_start_date", axis=1, inplace=True)

    # encode city as binary variable
    train_clean["city"] = train_clean["city"].map({"sj": 1, "iq": 0})
    test_clean["city"] = test_clean["city"].map({"sj": 1, "iq": 0})

    for i in test_clean.columns:
        test_clean[i] = test_clean[i].interpolate()

    if export:
        train_clean.to_csv("train_" + out_path)
        test_clean.to_csv("test_" + out_path)

    return train_clean, test_clean


def mean_encode(data, col, on):
    group = data.groupby(col).mean()
    mapper = {k: v for k, v in zip(group.index, group.loc[:, on].values)}

    with open("week_mapper.json", "w+") as file:
        file.write(json.dumps(mapper))

    data.loc[:, col] = data.loc[:, col].replace(mapper)
    data.loc[:, col].fillna(value=np.mean(data.loc[:, col]), inplace=True)

    return data


def add_population_data(
    df, iq_csv="./data/population_iq.csv", sj_csv="./data/population_sj.csv"
):
    pop_data_iq = pd.read_csv(iq_csv)
    pop_data_iq["city"] = 0
    pop_data_sj = pd.read_csv(sj_csv)
    pop_data_sj["city"] = 1
    df = pd.merge(df, pop_data_iq, how="left", on=["city", "year"])
    df = pd.merge(df, pop_data_sj, how="left", on=["city", "year"])
    df["population_x"] = df["population_x"].fillna(df["population_y"])
    df.drop("population_y", axis=1, inplace=True)

    return df


def add_temp_dew(df):
    df["temp_dew"] = df["reanalysis_dew_point_temp_k"] > df["station_min_temp_c"]
    return df


def add_lag_cols(df, cols_to_lag=["temp_dew"], lags=4):
    for i in cols_to_lag:
        df[i + "_l" + str(lags)] = df[i].shift(lags)
    df[i + "_l" + str(lags)] = df[i + "_l" + str(lags)].fillna(method="bfill")

    return df


def encode_weeks(df):

    yearly_cases = df.groupby("year").sum()["total_cases"]

    df = pd.merge(
        df, yearly_cases, how="left", on="year", suffixes=["_weekly", "_yearly"]
    )
    df["pct_cases"] = df["total_cases_weekly"] / df["total_cases_yearly"]
    print(df.columns)
    mean_encode(df, "weekofyear", "pct_cases")

    return df


def export_submission(test_clean, test_processed, out_path, model):

    test_clean["total_cases"] = model.predict(test_processed)
    test_clean["total_cases"] = test_clean["total_cases"].astype(int)
    submission = test_clean[["city", "year", "weekofyear", "total_cases"]]
    submission["city"] = submission["city"].map({1: "sj", 0: "iq"})
    submission.to_csv(out_path, index=False)

    return submission


def preprocess_old(train_features, test_features):
    # convert kelvin to celsius
    train_features[
        [
            "reanalysis_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_dew_point_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_tdtr_k",
        ]
    ] = (
        train_features[
            [
                "reanalysis_air_temp_k",
                "reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k",
                "reanalysis_max_air_temp_k",
                "reanalysis_min_air_temp_k",
                "reanalysis_tdtr_k",
            ]
        ]
        - 273.15
    )
    test_features[
        [
            "reanalysis_air_temp_k",
            "reanalysis_avg_temp_k",
            "reanalysis_dew_point_temp_k",
            "reanalysis_max_air_temp_k",
            "reanalysis_min_air_temp_k",
            "reanalysis_tdtr_k",
        ]
    ] = (
        test_features[
            [
                "reanalysis_air_temp_k",
                "reanalysis_avg_temp_k",
                "reanalysis_dew_point_temp_k",
                "reanalysis_max_air_temp_k",
                "reanalysis_min_air_temp_k",
                "reanalysis_tdtr_k",
            ]
        ]
        - 273.15
    )

    # dropping rows where no temperature data is available

    train_clean = train_features.dropna(
        subset=["reanalysis_air_temp_k", "reanalysis_avg_temp_k", "station_avg_temp_c"],
        how="all",
    )

    # inputation for submission
    na_col = ["reanalysis_air_temp_k", "reanalysis_avg_temp_k", "station_avg_temp_c"]

    test_clean = test_features.copy(deep=True)

    # inputation - temperature data

    train_clean["station_avg_temp_c"].fillna(
        train_clean["reanalysis_avg_temp_k"], inplace=True
    )
    train_clean["station_diur_temp_rng_c"].fillna(
        train_clean["reanalysis_tdtr_k"], inplace=True
    )
    train_clean["station_max_temp_c"].fillna(
        train_clean["reanalysis_max_air_temp_k"], inplace=True
    )
    train_clean["station_min_temp_c"].fillna(
        train_clean["reanalysis_min_air_temp_k"], inplace=True
    )

    test_clean["station_avg_temp_c"].fillna(
        test_clean["reanalysis_avg_temp_k"], inplace=True
    )
    test_clean["station_diur_temp_rng_c"].fillna(
        test_clean["reanalysis_tdtr_k"], inplace=True
    )
    test_clean["station_max_temp_c"].fillna(
        test_clean["reanalysis_max_air_temp_k"], inplace=True
    )
    test_clean["station_min_temp_c"].fillna(
        test_clean["reanalysis_min_air_temp_k"], inplace=True
    )

    # inputation - vegetation index

    for i in ["ndvi_ne", "ndvi_sw", "ndvi_nw", "ndvi_se"]:
        train_clean[i] = train_clean[i].interpolate()
        test_clean[i] = test_clean[i].interpolate()

    # inputation - precipitation level
    train_clean["station_precip_mm"].fillna(
        train_clean["reanalysis_sat_precip_amt_mm"], inplace=True
    )
    test_clean["station_precip_mm"].fillna(
        test_clean["reanalysis_sat_precip_amt_mm"], inplace=True
    )

    # dropping duplicate columns
    train_clean.drop(
        ["precipitation_amt_mm", "reanalysis_sat_precip_amt_mm"], axis=1, inplace=True
    )
    test_clean.drop(
        ["precipitation_amt_mm", "reanalysis_sat_precip_amt_mm"], axis=1, inplace=True
    )

    # drop useless column
    train_clean.drop("week_start_date", axis=1, inplace=True)
    test_clean.drop("week_start_date", axis=1, inplace=True)

    # encode city as binary variable
    train_clean["city"] = train_clean["city"].map({"sj": 1, "iq": 0})
    test_clean["city"] = test_clean["city"].map({"sj": 1, "iq": 0})

    for i in test_clean.columns:
        test_clean[i] = test_clean[i].interpolate()

    return train_clean, test_clean


def get_features(opt):

    # choose features
    if opt == "AvgTemp_Prec":
        features = ["station_avg_temp_c", "precipitation_amt_mm"]
    elif opt == "AvgTemp_Prec_NDVI":
        features = ["station_avg_temp_c", "precipitation_amt_mm", "ndvi_ne"]
    else:
        raise (ValueError(f"features:{opt} option not defined"))

    return features
