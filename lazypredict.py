# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# !pip install lazypredict
from lazypredict.Supervised import LazyRegressor

from generator import TimeSeriesDataGenerator





def preprocessing(
    data: pd.DataFrame,
    window: int = 30,
    split_ratio: list = [0.8, 0.2],
) -> tuple:
    # Sliding
    df = pd.DataFrame(index=data.index)
    for i in range(window + 1):
        df[i] = data['value'].shift(window - i)
    df.dropna(inplace=True)

    # Spliting
    _ratio = [i/sum(split_ratio) for i in split_ratio]
    train_X, test_X = train_test_split(df, train_size=_ratio[0], test_size=_ratio[1], shuffle=False)
    train_Y = train_X.pop(train_X.columns[-1])
    test_Y = test_X.pop(test_X.columns[-1])

    return train_X, train_Y, test_X, test_Y






if __name__ == "__main__":
    date_range = {
        'start_time': '2024-01-01 00:00:00',
        'end_time': '2024-01-31 23:59:59',
        'freq': 'min',
    }

    clipping_range = {
        'min_bound': 0.,
        'max_bound': 500.,
    }

    pattern = {
        'Basic': 5,        # -1(None, Default), 1(Linear), 2(Sin), 3(Uniform), 4(Normal), 5(Server), 6(User)
        'Vibration': 1,    # -1(None, Default), 1(Noraml), 2(Uniform)
        'Abnoraml': 1,     # -1(None, Default), 1(1% Point Only), 2(2.5% Point, 2.5% Pattern),
                           # 3(5% Point, 5% Pattern), 4(12.5% Point, 12.5% Pattern), 5(25% Point, 25% Pattern)
    }

    seed = 42
    window = 30
    split_ratio = [8, 2]


    generator = TimeSeriesDataGenerator(date_range=date_range, clipping_range=clipping_range)
    data, abnormal_label = generator.run(pattern=pattern, abnormal=True, plot=True, seed=42)
    data = data.set_index(["datetime"])

    train_X, train_Y, test_X, test_Y = preprocessing(data, window, split_ratio)
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    X, y = shuffle(train_X, train_Y, random_state=seed)

    reg = LazyRegressor()
    models, predictions = reg.fit(
        X_train=X.values.astype(np.float32),
        y_train=y.values.astype(np.float32),
        X_test=test_X.values.astype(np.float32),
        y_test=test_Y.values.astype(np.float32),
    )
    print(models)

    model_name = "HistGradientBoostingRegressor"
    res = pd.DataFrame(index=test_X.index)
    res["real"] = test_Y.values
    res["pred"] = reg.models[model_name].predict(test_X)
    print(res)
