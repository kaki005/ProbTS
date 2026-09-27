# ---------------------------------------------------------------------------------
# Portions of this file are derived from GluonTS
# - Source: https://github.com/awslabs/gluonts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Any

import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.transform import MapTransformation
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    日時インデックスから時間特徴量を計算する時間特徴量の基底クラス。
    """

    def __init__(self) -> None:
        """
        TimeFeature を初期化する。
        """
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから時間特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            時間特徴量を計算する対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            計算された時間特徴量 (基底クラスでは何も返さない)。
        """
        pass

    def __repr__(self) -> str:
        """
        オブジェクトの文字列表現を返す。

        Returns:
        ----------
        str
            "クラス名()" 形式の文字列。
        """
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    分内の秒を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから分内の秒を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた分内の秒の特徴量。
        """
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    時内の分を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから時内の分を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた時内の分の特徴量。
        """
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    日内の時を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから日内の時を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた日内の時の特徴量。
        """
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    週内の曜日を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから週内の曜日を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた週内の曜日の特徴量。
        """
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    月内の日を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから月内の日を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた月内の日の特徴量。
        """
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    年内の日を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから年内の日を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた年内の日の特徴量。
        """
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    年内の月を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから年内の月を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた年内の月の特徴量。
        """
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    年内の週を [-0.5, 0.5] の範囲の値にエンコードする時間特徴量。
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスから年内の週を [-0.5, 0.5] の範囲にエンコードした特徴量を計算する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            [-0.5, 0.5] の範囲にエンコードされた年内の週の特徴量。
        """
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> list[TimeFeature]:
    """
    与えられた頻度文字列に適した時間特徴量のリストを返す。

    Parameters:
    ----------
    freq_str : str
        "12H", "5min", "1D" などの [倍数][粒度] 形式の頻度文字列。

    Returns:
    ----------
    list[TimeFeature]
        頻度に適した時間特徴量インスタンスのリスト。

    Raises:
    ----------
    RuntimeError
        サポートされていない頻度が指定された場合。
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = "h") -> np.ndarray:
    """
    頻度に基づく時間特徴量を計算し、縦に積み重ねた配列として返す。

    Parameters:
    ----------
    dates : pd.DatetimeIndex
        時間特徴量を計算する対象の日時インデックス。
    freq : str, optional, default="h"
        データの頻度文字列。

    Returns:
    ----------
    np.ndarray
        形状 (特徴量数, 時点数) の時間特徴量配列。
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


class FourierDateFeatures(TimeFeature):
    """
    周期的な日時属性を cos / sin によるフーリエ特徴量としてエンコードする時間特徴量。

    Attributes:
    ----------
    freq : str
        エンコード対象の日時属性名 (例: 'month', 'hour', 'dayofweek')。
    """

    freq: str

    def __init__(self, freq: str) -> None:
        """
        FourierDateFeatures を初期化する。

        Parameters:
        ----------
        freq : str
            エンコード対象の日時属性名。'month', 'day', 'hour', 'minute', 'weekofyear',
            'weekday', 'dayofweek', 'dayofyear', 'daysinmonth' のいずれか。
        """
        super().__init__()
        # reocurring freq
        freqs = [
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        日時インデックスの属性値を cos / sin のフーリエ特徴量に変換する。

        Parameters:
        ----------
        index : pd.DatetimeIndex
            対象の日時インデックス。

        Returns:
        ----------
        np.ndarray
            形状 (2, 時点数) の配列 (1 行目が cos、2 行目が sin)。
        """
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def norm_freq_str(freq_str: str) -> str:
    """
    頻度文字列を正規化し、基本頻度を返す (開始頻度を示す末尾の 'S' を除去する)。

    Parameters:
    ----------
    freq_str : str
        正規化する頻度文字列 (例: 'AS-JAN', 'H')。

    Returns:
    ----------
    str
        正規化された基本頻度文字列。
    """
    base_freq = freq_str.split("-")[0]

    # Pandas has start and end frequencies, e.g `AS` and `A` for yearly start
    # and yearly end frequencies. We don't make that difference and instead
    # rely only on the end frequencies which don't have the `S` prefix.
    # Note: Secondly ("S") frequency exists, where we don't want to remove the
    # "S"!
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]

    return base_freq


def fourier_time_features_from_frequency(freq_str: str) -> list[TimeFeature]:
    """
    与えられた頻度文字列に適したフーリエ時間特徴量のリストを返す。

    Parameters:
    ----------
    freq_str : str
        頻度文字列 (例: 'H', 'D', 'min')。

    Returns:
    ----------
    list[TimeFeature]
        頻度に適した FourierDateFeatures インスタンスのリスト。
    """
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)
    granularity = granularity.upper()
    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: list[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes


def get_lags(freq_str: str) -> list[int]:
    """
    データの頻度に基づいて、時系列予測に適したラグ値を計算する。

    Parameters:
    ----------
    freq_str : str
        時系列データの頻度。'M', 'D', 'B', 'H', 'T' / 'min' などをサポートし、
        それ以外の場合は [1] を返す。

    Returns:
    ----------
    list[int]
        モデルに含める過去の観測値のオフセットを表すラグ値のリスト。
        指定された頻度に対する自己相関や季節性パターンを捉えるように調整されている。

    Examples:
    ----------
    >>> get_lags("H")
    [1, 24, 168]  # 毎時・日次・週次の季節性を捉える

    >>> get_lags("D")
    [1, 7, 14]  # 日次・週次・隔週の季節性を捉える
    """
    freq_str = freq_str.upper()
    if freq_str == "M":
        lags = [1, 12]
    elif freq_str == "D":
        lags = [1, 7, 14]
    elif freq_str == "B":
        lags = [1, 2]
    elif freq_str == "H":
        lags = [1, 24, 168]
    elif freq_str in ("T", "min"):
        lags = [1, 4, 12, 24, 48]
    else:
        lags = [1]

    return lags


def target_transformation_length(
    target: np.ndarray, pred_length: int, is_train: bool
) -> int:
    """
    変換後の特徴量の長さを計算する。

    Parameters:
    ----------
    target : np.ndarray
        時系列の値を含むターゲット配列 (最後の次元が時間方向)。
    pred_length : int
        予測長。
    is_train : bool
        学習時かどうか。True の場合はターゲット長、False の場合はターゲット長 + pred_length。

    Returns:
    ----------
    int
        変換後の特徴量の長さ。
    """
    return target.shape[-1] + (0 if is_train else pred_length)


class AddCustomizedTimeFeatures(MapTransformation):
    """
    事前計算済みの時間特徴量をデータエントリに付与する変換。

    `is_train=True` の場合、特徴量行列は `target` フィールドと同じ長さになる。
    `is_train=False` の場合、特徴量行列の長さは `len(target) + pred_length` になる。

    Attributes:
    ----------
    date_features : np.ndarray
        事前計算済みの時間特徴量 (形状は (時点数, 特徴量数) など)。
    pred_length : int
        予測長。
    start_field : str
        時系列の開始タイムスタンプを持つフィールド名。
    target_field : str
        時系列の値の配列を持つフィールド名。
    output_field : str
        結果を格納するフィールド名。
    dtype : type
        2 次元特徴量の場合に用いるデータ型。
    """

    date_features: np.ndarray
    pred_length: int
    start_field: str
    target_field: str
    output_field: str
    dtype: type

    @validated()
    def __init__(
        self,
        start_field: str,
        target_field: str,
        output_field: str,
        time_features: Any,
        pred_length: int,
        dtype: type = np.float32,
    ) -> None:
        """
        AddCustomizedTimeFeatures を初期化する。

        Parameters:
        ----------
        start_field : str
            時系列の開始タイムスタンプを持つフィールド名。
        target_field : str
            時系列の値の配列を持つフィールド名。
        output_field : str
            結果を格納するフィールド名。
        time_features : Any
            使用する事前計算済みの時間特徴量 (np.ndarray)。
        pred_length : int
            予測長。
        dtype : type, optional, default=np.float32
            2 次元特徴量の場合に用いるデータ型。
        """
        self.date_features = time_features
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        """
        データエントリに時間特徴量を付与する。

        Parameters:
        ----------
        data : DataEntry
            変換対象のデータエントリ。
        is_train : bool
            学習時かどうか (特徴量の長さの決定に使用)。

        Returns:
        ----------
        DataEntry
            output_field に転置された時間特徴量が追加されたデータエントリ。
        """
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if len(self.date_features.shape) == 2:
            data[self.output_field] = self.date_features[:length].astype(self.dtype)
        else:
            data[self.output_field] = self.date_features[:length].astype(np.float64)
        data[self.output_field] = self.date_features[:length].astype(np.float64)
        data[self.output_field] = np.transpose(data[self.output_field])

        return data
