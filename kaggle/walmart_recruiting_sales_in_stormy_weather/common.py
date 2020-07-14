import numpy as np
import pandas as pd

TARGET_NAME = 'units'

def get_score(y_true, y_predict):
    return math.sqrt(mean_squared_error(y_true, y_predict))


def convert_to_minutes(x):
    x.loc[x == "-"] = np.nan
    x.loc[~x.isnull()] = (x[~x.isnull()].astype(int) % 100) + (x[~x.isnull()].astype(int) // 100 * 60)
    x = x.astype(float)

def get_unclaimed_products(dataset):
    product = np.unique(dataset["product"])
    all_non_empty_product = dataset[dataset["units"] > 0]["product"]
    return set(product) - set(all_non_empty_product)

def add_id_field(dataset):
    dataset["product"] = (dataset["store_nbr"].astype("str") + "_" + dataset["item_nbr"].astype("str")).to_numpy()
    dataset["id"] = (dataset["store_nbr"].astype("str") + "_" + dataset["item_nbr"].astype("str") + "_" + dataset["date"].astype('str')).to_numpy()


def preprocessing(data, key, weather, target_field=TARGET_NAME):
    data = data.join(key.set_index("store_nbr"), on="store_nbr")
    data = data.join(weather.set_index(["station_nbr", "date"]), on=["station_nbr", "date"])

    data.fillna(-1, inplace=True)

    result = data
    #result = pd.get_dummies(data, columns=["store_nbr", "item_nbr"])
    if target_field in data.columns:
        result.loc[:, target_field] = np.log(data[target_field] + 1)

    datetime_values = pd.to_datetime(data["date"], format="%Y-%m-%d")
    result.loc[:, "timestamp"] = datetime_values.values.astype(np.int64) // 1e9
    result.loc[:, "year"] = datetime_values.dt.year
    result.loc[:, "month"] = datetime_values.dt.month
    result.loc[:, "dayofweek"] = datetime_values.dt.dayofweek
    result.loc[:, "dayofmonth"] = datetime_values.dt.day
    result = result.drop(columns=["date", "codesum", "product", "id"])
    return result

