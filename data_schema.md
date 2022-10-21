# Data

## Training Dataset - Exploratory Analysis

| col name | type | value count | missing values |
| -------- | ---- | ----------- | -------------- | 
| city | object | 2 | 0 |
| year | int64 | 21 | 0 |
| weekofyear | int64 | 53 | 0 |
| week_start_date | object | 1049 | 0 |
| ndvi_ne | float64 | 1214 | 194 |
| ndvi_nw | float64 | 1365 | 52 |
| ndvi_se | float64 | 1395 | 22 |
| ndvi_sw | float64 | 1388 | 22 |
| precipitation_amt_mm | float64 | 1157 | 13 |
| reanalysis_air_temp_k | float64 | 1176 | 10 |
| reanalysis_avg_temp_k | float64 | 600 | 10 |
| reanalysis_dew_point_temp_k | float64 | 1180 | 10 |
| reanalysis_max_air_temp_k | float64 | 141 | 10 |
| reanalysis_min_air_temp_k | float64 | 117 | 10 |
| reanalysis_precip_amt_kg_per_m2 | float64 | 1039 | 10 |
| reanalysis_relative_humidity_percent | float64 | 1370 | 10 |
| reanalysis_sat_precip_amt_mm | float64 | 1157 | 13 |
| reanalysis_specific_humidity_g_per_kg | float64 | 1171 | 10 |
| reanalysis_tdtr_k | float64 | 519 | 10 |
| station_avg_temp_c | float64 | 492 | 43 |
| station_diur_temp_rng_c | float64 | 470 | 43 |
| station_max_temp_c | float64 | 73 | 20 |
| station_min_temp_c | float64 | 73 | 14 |
| station_precip_mm | float64 | 663 | 22 |

## Variable Description

- city – City abbreviations: sj for San Juan and iq for Iquitos
- week_start_date – Date given in yyyy-mm-dd format
- station_max_temp_c – Maximum temperature
- station_min_temp_c – Minimum temperature
- station_avg_temp_c – Average temperature
- station_precip_mm – Total precipitation
- station_diur_temp_rng_c – Diurnal temperature range
- precipitation_amt_mm – Total precipitation
- reanalysis_sat_precip_amt_mm – Total precipitation
- reanalysis_dew_point_temp_k – Mean dew point temperature
- reanalysis_air_temp_k – Mean air temperature
- reanalysis_relative_humidity_percent – Mean relative humidity
- reanalysis_specific_humidity_g_per_kg – Mean specific humidity
- reanalysis_precip_amt_kg_per_m2 – Total precipitation
- reanalysis_max_air_temp_k – Maximum air temperature
- reanalysis_min_air_temp_k – Minimum air temperature
- reanalysis_avg_temp_k – Average air temperature
- reanalysis_tdtr_k – Diurnal temperature range
- ndvi_se – Pixel southeast of city centroid
- ndvi_sw – Pixel southwest of city centroid
- ndvi_ne – Pixel northeast of city centroid
- ndvi_nw – Pixel northwest of city centroid
