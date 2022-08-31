
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


METRICS_COLS = ['impression_count', 'click_count', 'booking_count', 'avg_cpc', 'avg_clicked_price', 'avg_length_of_stay', 'avg_time_to_travel']
FEATURES_COLS = METRICS_COLS + ['stars', 'hotel_rating']

def clip_outliers(feature: pd.Series, lower_perc: int=1, upper_perc: int=98) -> np.ndarray:
    lower_bound, upper_bound = np.percentile(feature.values, [lower_perc, upper_perc])
    return np.clip(feature, lower_bound, upper_bound)

def fill_missing_rating(hotel_details: pd.DataFrame, stars2rating: Optional[dict] = None, special_char: str = '\\N') -> pd.DataFrame:
    if stars2rating is None:
        stars_ratings = hotel_details[hotel_details['hotel_rating'] >= 0].groupby(by=['stars'], as_index=False)['hotel_rating'].mean().to_frame()
        stars2rating = stars_ratings['mean'].to_dict()

    df_corrected = hotel_details.replace(special_char, -10)
    df_corrected['hotel_rating'] = df_corrected['hotel_rating'].astype('float')
    df_corrected.loc[df_corrected['hotel_rating'] < 0, ['hotel_rating']] = df_corrected[df_corrected['hotel_rating'] < 0]['stars'].apply(lambda star: stars2rating[star])
    
    return df_corrected

def create_features_df(
    hotel_metrics: pd.DataFrame,
    hotel_details: pd.DataFrame,
    feature_agg_columns: list,
    data_agg_metrics: Optional[pd.DataFrame] = None,
    city_agg_metrics: Optional[pd.DataFrame] = None,
    missing_city_kw: str = 'unknown',
) -> pd.DataFrame:
    
    united_hotel_metrics = pd.merge(hotel_metrics, hotel_details, how='left', on='hotel_id')
    united_hotel_metrics.drop(columns=['date_ymd', 'country'], inplace=True)
    
    if data_agg_metrics is None:
        data_agg_metrics = united_hotel_metrics[united_hotel_metrics['city'] != missing_city_kw][feature_agg_columns].mean().to_frame()
    
    if city_agg_metrics is None:
        city_agg_metrics = united_hotel_metrics[united_hotel_metrics['city'] != missing_city_kw].groupby(by=['city'], as_index=False)[feature_agg_columns].mean()
    
    cities = city_agg_metrics['city'].values

    for feature in feature_agg_columns:
        united_hotel_metrics[f'{feature}_all_ratio'] = united_hotel_metrics[feature] / data_agg_metrics.loc[feature].values[0]
        for city in cities:
            united_hotel_metrics[f'{feature}_{city.split()[0].lower()}_ratio'] = united_hotel_metrics[feature] / city_agg_metrics.loc[city_agg_metrics['city'] == city][feature].values[0]
    
    united_hotel_metrics.drop(columns=['hotel_id', 'city'], inplace=True)
    return united_hotel_metrics

def run_preprocess_pipeline(
    hotel_metrics: pd.DataFrame,
    hotel_details: pd.DataFrame,
    data_agg_metrics: Optional[pd.DataFrame] = None,
    city_agg_metrics: Optional[pd.DataFrame] = None,
    stars2missing_rating: Optional[dict] = None
) -> np.array:
    
    for col in METRICS_COLS:
        hotel_metrics[col] = clip_outliers(hotel_metrics[col])

    hotel_details = fill_missing_rating(hotel_details, stars2missing_rating)

    new_features = create_features_df(hotel_metrics, hotel_details, FEATURES_COLS, data_agg_metrics, city_agg_metrics)
    
    return scale(new_features)

def run_preprocess_pipeline_with_paths(
    hotel_metrics_path: str,
    hotel_details_path: str,
    data_agg_metrics_path: Optional[str] = None,
    city_agg_metrics_path: Optional[str] = None,
    stars2rating_path: Optional[str] = None
) -> np.array:
    '''
    Return features for CatBoost model inference given hotel metrics and pre-calculated statistics.
    If the paths to statistics are not provided, they will be recalculated on provided hotel data.
    '''
    
    hotel_details = pd.read_csv(hotel_details_path)
    hotel_metrics = pd.read_csv(hotel_metrics_path, parse_dates=['date_ymd'])
    
    reloaded_city_agg = pd.read_csv(city_agg_metrics_path)
    reloaded_data_agg = pd.read_csv(data_agg_metrics_path, index_col='metric')

    reloaded_stars_ratings = pd.read_csv(stars2rating_path, index_col='stars')
    stars2missing_rating = reloaded_stars_ratings['mean'].to_dict()

    return run_preprocess_pipeline(hotel_metrics, hotel_details, reloaded_data_agg, reloaded_city_agg, stars2missing_rating)
