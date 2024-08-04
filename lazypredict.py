# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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


def get_result(
    test_X: pd.DataFrame,
    test_Y: pd.DataFrame,
    reg: LazyRegressor,
    window: int = 30,
    model_name: str = "LinearRegression",
    minutes: list = [1, 5, 15, 30],
):
    # get predict result
    res = test_X.copy()
    for i in range(window):
        res[f"pred_{i + 1}"] = reg.models[model_name].predict(res.iloc[:, -30:].values)
    res = res.iloc[:, -30:]

    res["real_1"] = test_Y.values
    for i in range(2, window + 1):
        res[f"real_{i}"] = res.iloc[:, -1].shift(-1)
    res.dropna(inplace=True)

    # get figure
    fig, axes = plt.subplots(len(minutes), 2, figsize=(20, 5 * len(minutes)))

    fig.suptitle(f'{model_name} Model Time Series Prediction \n', fontsize=20)

    axes[0, 0].set_title('Entire Data', fontsize=15)
    axes[0, 1].set_title('Last 1 Day Data', fontsize=15)
    for i, minute in enumerate(minutes):
        axes[i, 0].set_ylabel(f'Predict {minute} Min After', labelpad=15, fontsize=15)

        axes[i, 0].plot(res[[f"real_{minute}", f"pred_{minute}"]], label=["real", "pred"])
        axes[i, 0].legend(loc='upper left', fontsize=10)

        axes[i, 1].plot(res[[f"real_{minute}", f"pred_{minute}"]].iloc[-24 * 60:], label=["real", "pred"])
        axes[i, 1].legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()

    return None





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


    model_name = "LinearRegression"

    res = pd.DataFrame(index=test_X.index)
    res["real"] = test_Y.values
    res["pred"] = reg.models[model_name].predict(test_X)
    print(res)


    minutes = [1, 5, 15, 30]
    get_result(test_X, test_Y, reg, window, model_name, minutes)
