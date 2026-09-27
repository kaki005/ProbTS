# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import numpy as np
from gluonts.time_feature import get_seasonality


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    平均二乗誤差 (MSE) を計算する。

    .. math::

        mse = mean((Y - \hat{Y})^2)

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        予測値 (target と同じ形状)。

    Returns:
    ----------
    float
        MSE の値。
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    絶対誤差の総和を計算する。

    .. math::

        abs\_error = sum(|Y - \hat{Y}|)

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        予測値 (target と同じ形状)。

    Returns:
    ----------
    float
        絶対誤差の総和。
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum(target: np.ndarray) -> float:
    r"""
    正解値の絶対値の総和を計算する。

    .. math::

        abs\_target\_sum = sum(|Y|)

    Parameters:
    ----------
    target : np.ndarray
        正解値。

    Returns:
    ----------
    float
        正解値の絶対値の総和。
    """
    return np.sum(np.abs(target))


def abs_target_mean(target: np.ndarray) -> float:
    r"""
    正解値の絶対値の平均を計算する。

    .. math::

        abs\_target\_mean = mean(|Y|)

    Parameters:
    ----------
    target : np.ndarray
        正解値。

    Returns:
    ----------
    float
        正解値の絶対値の平均。
    """
    return np.mean(np.abs(target))


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: np.ndarray,
) -> float:
    r"""
    平均絶対スケール誤差 (MASE) を計算する。

    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    詳細は [HA21]_ を参照。seasonal_error が 0 の要素では MASE を 0 とする。

    Parameters:
    ----------
    target : np.ndarray
        正解値 (masked array を想定)。形状 (batch_size, prediction_length, target_dim)。
    forecast : np.ndarray
        予測値 (target と同じ形状)。
    seasonal_error : np.ndarray
        季節性ナイーブ予測の誤差。形状 (batch_size, 1, target_dim)。

    Returns:
    ----------
    float
        MASE の値。
    """
    diff = np.mean(np.abs(target - forecast), axis=1)
    mase = diff / seasonal_error
    # if seasonal_error is 0, set mase to 0
    mase = mase.filled(0)
    return np.mean(mase)


def calculate_seasonal_error(
    past_data: np.ndarray,
    freq: str | None = None,
) -> np.ndarray:
    r"""
    過去データから季節性ナイーブ予測の誤差 (seasonal error) を計算する。

    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    ここで m は季節周期である。詳細は [HA21]_ を参照。
    季節周期が系列長以上の場合は m=1 にフォールバックする。

    Parameters:
    ----------
    past_data : np.ndarray
        過去の観測データ。形状 (batch_size, history_length, target_dim)。
    freq : str | None, optional, default=None
        データの頻度 (例: 'H', 'D')。季節周期の決定に使用される。

    Returns:
    ----------
    np.ndarray
        季節性誤差。形状 (batch_size, 1, target_dim)。
    """
    seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1

    y_t = past_data[:, :-forecast_freq]
    y_tm = past_data[:, forecast_freq:]

    mean_diff = np.mean(np.abs(y_t - y_tm), axis=1)
    mean_diff = np.expand_dims(mean_diff, axis=1)

    return mean_diff


def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    平均絶対パーセント誤差 (MAPE) を計算する。

    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    詳細は [HA21]_ を参照。

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        予測値 (target と同じ形状)。

    Returns:
    ----------
    float
        MAPE の値。
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    対称平均絶対パーセント誤差 (sMAPE) を計算する。

    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    詳細は [HA21]_ を参照。

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        予測値 (target と同じ形状)。

    Returns:
    ----------
    float
        sMAPE の値。
    """
    return 2 * np.mean(np.abs(target - forecast) / (np.abs(target) + np.abs(forecast)))


def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> np.ndarray:
    r"""
    要素ごとの分位点損失 (quantile loss) を計算する。

    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * ((Y <= \hat{Y}) - q)|)

    総和は取らず、要素ごとの値を返す (総和は呼び出し側で計算する)。

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        分位点 q における予測値 (target と同じ形状)。
    q : float
        分位点 (0 < q < 1)。

    Returns:
    ----------
    np.ndarray
        要素ごとの分位点損失 (target と同じ形状)。
    """
    return 2 * np.abs((forecast - target) * ((target <= forecast) - q))


def scaled_quantile_loss(
    target: np.ndarray, forecast: np.ndarray, q: float, seasonal_error: np.ndarray
) -> np.ndarray:
    """
    季節性誤差でスケーリングした分位点損失を計算する。

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        分位点 q における予測値 (target と同じ形状)。
    q : float
        分位点 (0 < q < 1)。
    seasonal_error : np.ndarray
        季節性ナイーブ予測の誤差 (スケーリング係数)。

    Returns:
    ----------
    np.ndarray
        スケーリングされた要素ごとの分位点損失。
    """
    return quantile_loss(target, forecast, q) / seasonal_error


def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    正解値が予測値を下回る割合 (coverage) を計算する。

    .. math::

        coverage = mean(Y < \hat{Y})

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    forecast : np.ndarray
        分位点における予測値 (target と同じ形状)。

    Returns:
    ----------
    float
        coverage の値。
    """
    return np.mean(target < forecast)
