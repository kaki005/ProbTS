import math
from copy import deepcopy
from datetime import datetime
from distutils.util import strtobool
from typing import Any

import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset, ListDataset
from gluonts.dataset.field_names import FieldName


def split_train_val(
    train_set: Dataset,
    num_test_windows: int,
    context_length: int,
    prediction_length: int,
    freq: str,
) -> tuple[ListDataset, ListDataset]:
    """
    学習データセットを、末尾を切り詰めた学習セットと検証セットに分割する。

    Parameters:
    ----------
    train_set : Dataset
        入力となる学習データセット。
    num_test_windows : int
        検証に用いるローリングウィンドウの数。
    context_length : int
        モデルのコンテキスト長。
    prediction_length : int
        モデルの予測ホライズン。
    freq : str
        データの頻度 (例: 'H' は毎時)。

    Returns:
    ----------
    tuple[ListDataset, ListDataset]
        切り詰めた学習データセット (trunc_train_set) と検証データセット (val_set) のタプル。
    """
    trunc_train_list = []
    val_set_list = []
    univariate = False

    for train_seq in iter(train_set):
        # truncate train set
        offset = num_test_windows * prediction_length
        trunc_train_seq = deepcopy(train_seq)

        if len(train_seq[FieldName.TARGET].shape) == 1:
            trunc_train_len = train_seq[FieldName.TARGET].shape[0] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][
                :trunc_train_len
            ]
            univariate = True
        elif len(train_seq[FieldName.TARGET].shape) == 2:
            trunc_train_len = train_seq[FieldName.TARGET].shape[1] - offset
            trunc_train_seq[FieldName.TARGET] = train_seq[FieldName.TARGET][
                :, :trunc_train_len
            ]
        else:
            raise ValueError(
                f"Invalid Data Shape: {len(train_seq[FieldName.TARGET].shape)!s}"
            )

        trunc_train_list.append(trunc_train_seq)

        # construct val set by rolling
        for i in range(num_test_windows):
            val_seq = deepcopy(train_seq)
            rolling_len = trunc_train_len + prediction_length * (i + 1)
            if univariate:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][
                    trunc_train_len
                    + prediction_length * (i - 1)
                    - context_length : rolling_len
                ]
            else:
                val_seq[FieldName.TARGET] = val_seq[FieldName.TARGET][:, :rolling_len]

            val_set_list.append(val_seq)

    trunc_train_set = ListDataset(
        trunc_train_list, freq=freq, one_dim_target=univariate
    )

    val_set = ListDataset(val_set_list, freq=freq, one_dim_target=univariate)

    return trunc_train_set, val_set


def truncate_test(
    test_set: Dataset, context_length: int, prediction_length: int, freq: str
) -> ListDataset:
    """
    テストデータセットを切り詰め、末尾のコンテキスト長と予測長の分のみを残す。

    Parameters:
    ----------
    test_set : Dataset
        入力となるテストデータセット。
    context_length : int
        モデルのコンテキスト長。
    prediction_length : int
        モデルの予測ホライズン。
    freq : str
        データの頻度。

    Returns:
    ----------
    ListDataset
        切り詰めたテストデータセット (trunc_test_set)。
    """
    trunc_test_list = []
    for test_seq in iter(test_set):
        # truncate train set
        trunc_test_seq = deepcopy(test_seq)

        trunc_test_seq[FieldName.TARGET] = trunc_test_seq[FieldName.TARGET][
            -(prediction_length * 2 + context_length) :
        ]

        trunc_test_list.append(trunc_test_seq)

    trunc_test_set = ListDataset(trunc_test_list, freq=freq, one_dim_target=True)

    return trunc_test_set


def get_rolling_test(
    stage: str,
    test_set: Dataset,
    border_begin_idx: int,
    border_end_idx: int,
    rolling_length: int,
    pred_len: int,
    freq: str | None = None,
) -> ListDataset:
    """
    ローリングウィンドウを用いてテストデータセットを構築する。

    Parameters:
    ----------
    stage : str
        ステージ名 (例: 'test', 'val')。
    test_set : Dataset
        テストデータセット (先頭の 1 系列のみを使用)。
    border_begin_idx : int
        ローリングウィンドウの開始インデックス。
    border_end_idx : int
        ローリングウィンドウの終了インデックス。
    rolling_length : int
        各ローリングウィンドウ間のずらし幅。
    pred_len : int
        予測長。
    freq : str | None, optional, default=None
        データの頻度。

    Returns:
    ----------
    ListDataset
        ローリングテストデータセット (rolling_test_set)。
    """
    num_test_windows = math.ceil(
        (border_end_idx - border_begin_idx - pred_len) / rolling_length
    )
    print(f"{stage}  pred_len: {pred_len} : num_test_windows: {num_test_windows}")

    test_set = next(iter(test_set))
    rolling_test_seq_list = list()
    for i in range(num_test_windows):
        rolling_test_seq = deepcopy(test_set)
        rolling_end = border_begin_idx + pred_len + i * rolling_length
        rolling_test_seq[FieldName.TARGET] = rolling_test_seq[FieldName.TARGET][
            :, :rolling_end
        ]
        rolling_test_seq_list.append(rolling_test_seq)

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=False
    )
    return rolling_test_set


def get_rolling_test_of_gift_eval(
    dataset: Dataset, prediction_length: int, windows: int
) -> ListDataset:
    """
    GiftEval 向けに、ローリングウィンドウを用いてテストデータセットを構築する。

    https://github.com/SalesforceAIResearch/gift-eval/blob/61ec5e563188bc4b2d7e86f6a7fcc78270607ae7/src/gift_eval/data.py#L213
    ウィンドウはデータセットの後方から取得する。例えばデータセットが N 個の時点を持つ場合:
    - 最初のウィンドウは最初の時点から N - prediction_length * windows 番目の時点まで。
    - 2 番目のウィンドウは最初の時点から N - prediction_length * (windows - 1) 番目の時点まで。
    - 最後のウィンドウは最初の時点から N 番目の時点まで。

    Parameters:
    ----------
    dataset : Dataset
        入力データセット (先頭の 1 系列のみを使用し、'freq' キーを含む必要がある)。
    prediction_length : int
        予測長。
    windows : int
        ローリングウィンドウの数。

    Returns:
    ----------
    ListDataset
        ローリングテストデータセット (rolling_test_set)。

    Raises:
    ----------
    ValueError
        データセットが 'freq' キーを含まない場合、またはターゲットの次元数が 1 / 2 以外の場合。
    """
    rolling_test_seq_list = list()
    dataset = next(iter(dataset))
    if "freq" not in dataset.keys():
        raise ValueError("The dataset must contain the 'freq' key.")
    freq = dataset["freq"]
    is_univariate = len(dataset[FieldName.TARGET].shape) == 1

    for i in range(windows):
        rolling_test_seq = deepcopy(dataset)
        rolling_end = dataset[FieldName.TARGET].shape[-1] - prediction_length * (
            windows - i
        )
        if is_univariate:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][:rolling_end]
        elif len(dataset[FieldName.TARGET].shape) == 2:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][
                :, :rolling_end
            ]
        else:
            raise ValueError(
                f"Invalid Data Shape: expected 1 or 2 dimensions, got {len(dataset[FieldName.TARGET].shape)}"
            )
        rolling_test_seq_list.append(rolling_test_seq)

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=is_univariate
    )
    return rolling_test_set


def df_to_mvds(df: pd.DataFrame, freq: str = "H") -> ListDataset:
    """
    pandas DataFrame を GluonTS 用の多変量 ListDataset に変換する。

    Parameters:
    ----------
    df : pd.DataFrame
        各列が時系列の変数を表す入力 DataFrame。
    freq : str, optional, default="H"
        データの頻度 (例: 'H' は毎時)。

    Returns:
    ----------
    ListDataset
        各変数を 1 系列とする多変量 ListDataset。
    """
    datasets = []
    for variable in df.keys():
        ds = {"item_id": variable, "target": df[variable], "start": str(df.index[0])}
        datasets.append(ds)
    dataset = ListDataset(datasets, freq=freq)
    return dataset


def convert_monash_data_to_dataframe(
    full_file_path_and_name: str,
    replace_missing_vals_with: Any = "NaN",
    value_column_name: str = "series_value",
) -> tuple[pd.DataFrame, str | None, int | None, bool | None, bool | None]:
    """
    Monash 形式 (.tsf) の時系列ファイルを読み込み、pandas DataFrame に変換する。

    Parameters:
    ----------
    full_file_path_and_name : str
        読み込む .tsf ファイルのパス。
    replace_missing_vals_with : Any, optional, default="NaN"
        欠損値 ("?") を置き換える値。
    value_column_name : str, optional, default="series_value"
        系列の値を格納する列名。

    Returns:
    ----------
    tuple[pd.DataFrame, str | None, int | None, bool | None, bool | None]
        読み込んだ DataFrame (loaded_data)、頻度 (frequency)、予測ホライズン (forecast_horizon)、
        欠損値を含むかどうか (contain_missing_values)、系列長が等しいかどうか
        (contain_equal_length) のタプル。メタデータが存在しない項目は None。

    Raises:
    ----------
    Exception
        ファイルの形式が不正な場合 (メタデータ、属性、データセクションの欠落など)。
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def monash_format_convert(
    loaded_data: pd.DataFrame, frequency: str | None, multivariate: bool
) -> pd.DataFrame:
    """
    Monash 形式から読み込んだ DataFrame を、ProbTS で扱う形式の DataFrame に変換する。

    Parameters:
    ----------
    loaded_data : pd.DataFrame
        convert_monash_data_to_dataframe で読み込んだ DataFrame。
        'series_name', 'start_timestamp', 'series_value' 列を含む必要がある。
    frequency : str | None
        データの頻度 ("10_minutes", "daily" は pandas の頻度文字列に変換される)。
    multivariate : bool
        True の場合、'date' 列と各系列の列を持つ横持ちの DataFrame を返す。
        False の場合、各行が 1 系列 (target, start, feat_static_cat, item_id) の DataFrame を返す。

    Returns:
    ----------
    pd.DataFrame
        変換後の DataFrame。
    """
    series_names = loaded_data["series_name"].values

    if str(frequency) == "10_minutes":
        freq = "10min"
    elif str(frequency) == "daily":
        freq = "D"
    else:
        freq = frequency

    if multivariate:
        timestamps = pd.date_range(
            start=loaded_data["start_timestamp"][0],
            periods=len(loaded_data["series_value"][0]),
            freq=freq,
        )
        new_df = pd.DataFrame({"date": timestamps})

        series_df = pd.DataFrame(
            {
                series: loaded_data["series_value"][i]
                for i, series in enumerate(series_names)
            }
        )
        result_df = pd.concat([new_df, series_df], axis=1)
    else:
        result = []
        for idx, row in loaded_data.iterrows():
            result.append(
                {
                    "target": np.array(row["series_value"], dtype=np.float32),
                    "start": pd.Period(row["start_timestamp"], freq=freq),
                    "feat_static_cat": np.array([idx], dtype=np.int32),
                    "item_id": idx,
                }
            )
        result_df = pd.DataFrame(result)
    return result_df
