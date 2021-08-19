import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import SplineTransformer


def count_transformer():
    return FunctionTransformer(lambda x: x / 1000.0)


def weather_transformer():
    return FunctionTransformer(lambda x: "rain" if x == "heavy_rain" else x)


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True

    # Replace with KBinsDiscretizer
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


def working_day_tranformer():
    return FunctionTransformer(lambda x: x == "True")
