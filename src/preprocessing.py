import re
import numpy as np
import pandas as pd


loggias = ["лоджия застекленная", "лоджия", "лоджия из кухни застеклена",
           "лоджия из кухни", "лоджия из кухни застеклена + вагонка", "балкон и лоджия"]
balconies = ["балкон застекленный", "балкон", "балкон застекленный + вагонка"]
two_balconies = ["2 лоджии застекленные", "2 лоджии", "2 балкона", "2 балкона застекленные", "балкон+терраса", "3 лоджии застекленные",
                 "балкон+терраса", "2 лоджии застекленные + вагонка", "3 лоджии", "3 балкона", "2 балкона застекленные + вагонка",
                 "2 лоджии 1 застекленная", "3 балкона застекленных", "2 балкона 1 застекленный"]

house_type_map = {'каркасно-блочный': 'frame-block', 'панельный': 'panel', 'кирпичный': 'brick',
                  'монолитный': 'monolithic', 'блок-комнаты': 'block-rooms', 'силикатные блоки': 'silicate'}

map_bathroom = {'раздельный': 'separate', 'совмещенный': 'combined', '2 сан.узла': '2',
                '3 сан.узла': 'more than 2', '4 сан.узла': 'more than 2'}

map_district = {'Октябрьский': 'Oktyabrsky', 'Центральный': 'Tsentralny', 'Московский': 'Moskovsky',
                'Фрунзенский': 'Frunzensky', 'Первомайский': 'Pervomaisky', 'Советский': 'Sovetsky',
                'Ленинский': 'Leninsky', 'Заводской': 'Zavodskoy', 'Партизанский': 'Partizansky'}


def get_floor(x):
    split = str(x).split('/')
    return int(split[0].strip()) if len(split) > 1 else np.nan


def get_num_of_storeys(x):
    split = str(x).split('/')
    return int(split[1].strip()) if len(split) > 1 else np.nan


def get_ceiling_height(x):
    x = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
    return float(x[0]) if x else np.nan


def get_target_price(x):
    x = re.findall(r'\b\d+\b', str(x))
    return float(x[0])*1000 if x else np.nan


def get_total_area(x):
    areas = str(x).split('/')
    areas += [np.nan]*(3-len(areas))
    return float(areas[0].strip()) if str(areas[0].strip())[0].isnumeric() else np.nan


def get_living_area(x):
    areas = str(x).split('/')
    areas += [np.nan]*(3-len(areas))
    return float(areas[1].strip()) if str(areas[1].strip())[0].isnumeric() else np.nan


def get_kitchen_area(x):
    areas = str(x).split('/')
    areas += [np.nan]*(3-len(areas))
    return float(areas[2].split()[0]) if str(areas[2].split()[0].strip())[0].isnumeric() else np.nan


def map_balcony(x):
    if x == 'нет':
        return 'Without'
    elif x in loggias:
        return 'Loggia'
    elif x in balconies:
        return 'Balcony'
    elif x in two_balconies:
        return 'Two balconies'
    else:
        return np.nan


def clean_data(df):
    features = ['Этаж / этажность', 'Тип дома', 'Высота потолков', 'Метро',
                'Район города', 'Цена USD', 'Год постройки', 'Площадь общая/жилая/кухня',
                'Комнат всего/разд.', 'Балкон', 'Сан/узел']

    df = df[features]
    df = df.assign(
        floor = df['Этаж / этажность'].apply(get_floor),
        number_of_storeys = df['Этаж / этажность'].apply(get_num_of_storeys),
        ceiling_height = df['Высота потолков'].apply(get_ceiling_height),
        target_price = df['Цена USD'].apply(get_target_price),
        total_area = df['Площадь общая/жилая/кухня'].apply(get_total_area),
        living_area = df['Площадь общая/жилая/кухня'].apply(get_living_area),
        kitchen_area = df['Площадь общая/жилая/кухня'].apply(get_kitchen_area),
        district = df['Район города'].apply(lambda x: str(x).split()[0]).map(map_district),
        near_the_subway = df['Метро'].apply(lambda x: 'Yes' if x is not np.nan else 'No'),
        number_of_rooms = df['Комнат всего/разд.'].apply(lambda x: int(re.findall(r'\d+', x)[0])),
        house_type = df['Тип дома'].map(house_type_map),
        bathroom = df['Сан/узел'].map(map_bathroom),
        balcony = df['Балкон'].map(map_balcony),
        year_built = df['Год постройки']
    )
    df = df.drop(features, axis=1)
    return df


def remove_outliers(df):
    df = df[(df['total_area'] > 20) & (df['total_area'] < 500)]
    df = df[df['ceiling_height'] < 4]
    df = df[(df['number_of_rooms'] > 0) & (df['number_of_rooms'] < 10)]
    df = df[(df['target_price'] > 0) & (df['target_price'] < 500000)]
    return df


def handle_missing_values(df):
    df['floor'] = df['floor'].fillna(df['floor'].mode()[0])
    df['number_of_storeys'] = df['number_of_storeys'].fillna(df['number_of_storeys'].mode()[0])
    df['house_type'] = df['house_type'].fillna(df['house_type'].mode()[0])
    df['district'] = df['district'].fillna(df['district'].mode()[0])
    df['ceiling_height'] = df['ceiling_height'].fillna(df['ceiling_height'].median())
    df['bathroom'] = df['bathroom'].fillna(df['bathroom'].mode()[0])
    df['balcony'] = df['balcony'].fillna(df['balcony'].mode()[0])
    df = df.dropna()
    return df 


def preprocess_data(path):
    df = pd.read_csv(path)
    df = clean_data(df)
    df = remove_outliers(df)
    df = handle_missing_values(df)
    
    X = df.drop('target_price', axis=1)
    y = df['target_price']
    return X, y
