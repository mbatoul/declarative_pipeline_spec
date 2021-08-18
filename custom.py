import numpy as np


def rescale_target(y):
    return y / y.max()


def collapse_heavy_rain_into_rain(X):
    X["weather"].replace(to_replace="heavy_rain", value="rain", inplace=True)


def sin_transformer(x, period):
    return np.sin(x / period * 2 * np.pi)


def cos_transformer(x, period):
    return np.cos(x / period * 2 * np.pi)
