import json

import geopandas
import geopy.distance as geopyd
import joblib
import numpy as np
import pandas as pd
import polyline
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm.notebook import tqdm


def convert_dates(df, minutes_window=10):
    """Input: a dataframe with a selection time column to be converted
    Output: a dataframe with parsed dates and dropped initial columns
    """
    df['selection time'] = pd.to_datetime(df['selection time'])
    df['month'] = df['selection time'].dt.month
    df['weekday'] = df['selection time'].dt.weekday
    df['hour'] = df['selection time'].dt.hour
    df['minute'] = df['selection time'].dt.minute
    df['dayofyear'] = df['selection time'].dt.dayofyear
    df['minuteofday'] = minutes_window * \
        ((df['minute'] + 60 * df['hour']) // minutes_window)

    df['true hour'] = df['time key sélection'].apply(lambda x: x // 1e4)
    df['differentiel hour'] = ((df['true hour'] - df['hour']) % 24)

    df = df[df['differentiel hour'] <= 1]   # remove some outliers
    df['differentiel hour'] = df['differentiel hour'].astype(bool)
    df.drop(['minute', 'selection time', 'date key sélection',
             'time key sélection'], axis=1, inplace=True)

    return df


def dist(x):
    """Euclidean distance between coordinates"""
    return int(geopyd.distance((x[0], x[1]), (x[2], x[3])).m)


def add_distance(df):
    coords_columns = ['longitude intervention', 'latitude intervention',
                      'longitude before departure', 'latitude before departure']

    distances = joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
        joblib.delayed(dist)(point) for point in tqdm(df[coords_columns].values, leave=False))

    df['global_distance'] = np.array(distances).astype(int)
    return df


def parse_osrm(df):
    df['OSRM response parsed'] = df['OSRM response'].apply(json.loads)
    df['OSRM geometry decoded'] = df['OSRM response parsed'].apply(lambda x: polyline.decode(x['routes'][0]['geometry'],
                                                                                             precision=6,
                                                                                             geojson=True))

    df = df.drop('OSRM response', axis=1)
    return df


def distance_vol(points):
    return sum([geopyd.distance(points[i], points[i + 1]).m for i in range(len(points) - 1)])


def add_distance_vol(df):
    distances = joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
        joblib.delayed(distance_vol)(point) for point in tqdm(df['OSRM geometry decoded'].values, leave=False))

    df['OSRM distance vol'] = np.array(distances).astype(int)

    return df


def parse_GPS_tracks(df):
    df['GPS split'] = df['GPS tracks departure-presentation'].apply(
        lambda x: x.split(';') if isinstance(x, str) else np.nan)
    df['GPS lat-long'] = df['GPS split'].apply(lambda x: [list(
        map(float, f.split(','))) for f in x] if isinstance(x, list) else np.nan)

    df['GPS long'] = df['GPS lat-long'].apply(
        lambda x: np.array(x)[:, 1] if isinstance(x, list) else np.nan)
    df['GPS lat'] = df['GPS lat-long'].apply(
        lambda x: np.array(x)[:, 0] if isinstance(x, list) else np.nan)

    df = df.drop('GPS tracks departure-presentation', axis=1)

    return df


def make_vehicle_categories(df_train, df_test):
    df_all = pd.concat([df_train, df_test])

    with open('category_engins.json') as f:
        dict_engins = json.load(f)

    vehicle_types = pd.Series(list(
        set(map(lambda x: x.split()[0], df_all['emergency vehicle type'].unique()))))
    vehicle_types_match = vehicle_types.apply(lambda x: (
        x, [engin for engin in dict_engins if x in dict_engins[engin]]))
    vehicle_categ = {k: v[0] for k, v in vehicle_types_match}

    df_train['emergency vehicle category'] = df_train['emergency vehicle type'].apply(
        lambda x: vehicle_categ[x.split()[0]])
    df_test['emergency vehicle category'] = df_test['emergency vehicle type'].apply(
        lambda x: vehicle_categ[x.split()[0]])

    return df_train, df_test


def find_departement(x, polygons):
    if polygons[4].contains(x):
        return 4
    elif polygons[5].contains(x):
        return 5
    elif polygons[7].contains(x):
        return 7
    elif polygons[1].contains(x):
        return 1
    elif polygons[2].contains(x):
        return 2
    elif polygons[3].contains(x):
        return 3
    elif polygons[6].contains(x):
        return 6
    else:
        return -1


def add_departements(df):
    paris_path = 'shape_files/geoflar-departements/geoflar-departements.shp'
    paris = geopandas.read_file(paris_path)

    depart_cols = ['longitude before departure', 'latitude before departure']
    arrivee_cols = ['longitude intervention', 'latitude intervention']

    print('\t- Convert points')
    df['point depart'] = df[depart_cols] .apply(
        lambda x: Point(x[0], x[1]), axis=1)
    df['point arrivee'] = df[arrivee_cols].apply(
        lambda x: Point(x[0], x[1]), axis=1)

    print('\t- Find departements')
    df['departement depart'] = df['point depart'] .apply(
        lambda x: find_departement(x, paris['geometry'].values))
    df['departement arrivee'] = df['point arrivee'].apply(
        lambda x: find_departement(x, paris['geometry'].values))
    df['inter departement'] = (
        df['departement depart'] != df['departement arrivee'])

    return df


def winsorize_series(s, lower=0.0, upper=0.9):
    """Utility winsorization function"""
    q = s.quantile([lower, upper])
    if isinstance(q, pd.Series) and len(q) == 2:
        s[s < q.iloc[0]] = q.iloc[0]
        s[s > q.iloc[1]] = q.iloc[1]
    return s


def add_ratios(df, eps=1e-5):

    df['ratio estimation'] = df['OSRM estimated distance'] / \
        (df['OSRM estimated duration'] + eps)
    df['ratio new estimation'] = df['OSRM estimated distance from last observed GPS position'] / \
        (df['OSRM estimated duration from last observed GPS position'] + eps)

    df['ratio estimated vol'] = df['OSRM estimated distance'] / \
        (df['OSRM distance vol'] + eps)

    df['ratio distance vs new distance'] = df['OSRM estimated distance'] / \
        (df['OSRM estimated distance from last observed GPS position'] + eps)
    df['ratio duration vs new duration'] = df['OSRM estimated duration'] / \
        (df['OSRM estimated duration from last observed GPS position'] + eps)

    ratio_columns = ['ratio estimation', 'ratio new estimation', 'ratio estimated vol',
                     'ratio distance vs new distance', 'ratio duration vs new duration']

    for ratio_c in ratio_columns:
        df[ratio_c] = winsorize_series(
            df[ratio_c] * 100).fillna(-1).astype(int)

    return df, ratio_columns
