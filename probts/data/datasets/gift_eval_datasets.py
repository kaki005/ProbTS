# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections.abc import Iterable, Iterator
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose

import datasets

# add for probts transform
from probts.data.data_utils.data_utils import get_rolling_test_of_gift_eval

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
}


class Term(Enum):
    """
    予測期間 (ターム) の種類を表す列挙型。

    Attributes:
    ----------
    SHORT : str
        短期予測 ("short")。
    MEDIUM : str
        中期予測 ("medium")。
    LONG : str
        長期予測 ("long")。
    """

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        """
        ベース予測長に掛ける倍率を返す。

        Returns:
        ----------
        int
            SHORT なら 1、MEDIUM なら 10、LONG なら 15。
        """
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """
    データエントリの "start" フィールドを numpy スカラーから Python オブジェクトに変換する。

    Parameters:
    ----------
    data_entry : DataEntry
        変換対象のデータエントリ。

    Returns:
    ----------
    DataEntry
        "start" フィールドを .item() で変換したデータエントリ。
    """
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class MultivariateToUnivariate(Transformation):
    """
    多変量系列を次元ごとの単変量系列に分解する変換クラス。

    Attributes:
    ----------
    field : str
        分解対象のフィールド名 (例: "target")。
    """

    field: str

    def __init__(self, field: str) -> None:
        """
        MultivariateToUnivariate を初期化する。

        Parameters:
        ----------
        field : str
            分解対象のフィールド名 (例: "target")。
        """
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator[DataEntry]:
        """
        各データエントリを次元ごとに分解し、単変量エントリとして順に返す。
        item_id には "_dim{次元番号}" が付与される。

        Parameters:
        ----------
        data_it : Iterable[DataEntry]
            入力データエントリのイテラブル。
        is_train : bool, optional, default=False
            学習時かどうか (本変換では未使用)。

        Returns:
        ----------
        Iterator[DataEntry]
            単変量化されたデータエントリを返すイテレータ。
        """
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                data_entry[self.field] = val
                data_entry["item_id"] = item_id + "_dim" + str(id)
                yield data_entry


class GiftEvalDataset:
    """
    GIFT-Eval ベンチマークのデータセットを読み込み、学習・検証・テスト用に分割するクラス。

    Attributes:
    ----------
    term : Term
        予測期間 (ターム) の種類。
    name : str
        データセット名 (ストレージ上のディレクトリ名)。
    to_univariate : bool
        多変量系列を単変量系列に分解するかどうか。
    hf_dataset : datasets.Dataset
        ディスクから読み込んだ HuggingFace データセット (numpy 形式)。
    """

    term: Term
    name: str
    to_univariate: bool
    hf_dataset: "datasets.Dataset"

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        storage_env_var: str = "GIFT_EVAL",
    ) -> None:
        """
        GiftEvalDataset を初期化し、HuggingFace データセットをディスクから読み込む。

        Parameters:
        ----------
        name : str
            データセット名 (ストレージ上のディレクトリ名)。
        term : Term | str, optional, default=Term.SHORT
            予測期間 (ターム) の種類。
        to_univariate : bool, optional, default=False
            多変量系列を単変量系列に分解するかどうか。
        storage_env_var : str, optional, default="GIFT_EVAL"
            データセットの保存先パスを格納した環境変数名。
        """
        self.term = Term(term)
        self.name = name
        self.to_univariate = to_univariate

        load_dotenv()
        storage_path = Path(os.getenv(storage_env_var))
        self.hf_dataset = datasets.load_from_disk(str(storage_path / name)).with_format(
            "numpy"
        )

    @cached_property
    def gluonts_dataset(self) -> Any:
        """
        HuggingFace データセットを GluonTS 形式のデータセットに変換する。

        Returns:
        ----------
        Any
            GluonTS 形式のデータセット (Map、または to_univariate 時は変換済みデータセット)。
        """
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )
        gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if self.to_univariate:
            gluonts_dataset = MultivariateToUnivariate("target").apply(gluonts_dataset)
        return gluonts_dataset

    @cached_property
    def prediction_length(self) -> int:
        """
        データ頻度とタームに基づいて予測長を返す。

        Returns:
        ----------
        int
            予測長 (ベース予測長 × タームの倍率)。
        """
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        """
        データセットの頻度を返す。

        Returns:
        ----------
        str
            データ頻度の文字列。
        """
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        """
        ターゲットの次元数 (変数の数) を返す。

        Returns:
        ----------
        int
            ターゲットの次元数。単変量なら 1。
        """
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def target_ndim(self) -> int:
        """
        ターゲット配列の次元数を返す。

        Returns:
        ----------
        int
            単変量なら 1、多変量なら 2。
        """
        return 1 if self.target_dim == 1 else 2

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        """
        past_feat_dynamic_real の次元数を返す。

        Returns:
        ----------
        int
            past_feat_dynamic_real の次元数。存在しない場合は 0。
        """
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif (
            len(
                (
                    past_feat_dynamic_real := self.hf_dataset[0][
                        "past_feat_dynamic_real"
                    ]
                ).shape
            )
            > 1
        ):
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        """
        テスト用ローリングウィンドウ数を返す。

        Returns:
        ----------
        int
            ウィンドウ数 (m4 では 1、それ以外は 1 以上 MAX_WINDOW 以下)。
        """
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        """
        データセット内の系列長の最小値を返す。

        Returns:
        ----------
        int
            系列長の最小値。
        """
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        """
        データセット内の系列長の合計を返す。

        Returns:
        ----------
        int
            系列長の合計。
        """
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column("target"))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        """
        学習用データセットを返す。

        Returns:
        ----------
        TrainingDataset
            末尾から prediction_length * (windows + 1) を除いた学習用データセット。
        """
        training_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1)
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        """
        検証用データセットを返す。

        Returns:
        ----------
        TrainingDataset
            末尾から prediction_length * windows を除いた検証用データセット。
        """
        validation_dataset, _ = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        return validation_dataset

    @property
    def test_dataset(self) -> TrainingDataset:
        """
        ローリングウィンドウで構築したテスト用データセットを返す (ベータ版)。

        Returns:
        ----------
        TrainingDataset
            ローリングテスト用データセット (ListDataset)。
        """
        print(
            f"BETA version: generating test datasets for gift eval, should contain {self.windows} windows."
        )
        test_dataset = get_rolling_test_of_gift_eval(
            dataset=self.gluonts_dataset,
            prediction_length=self.prediction_length,
            windows=self.windows,
        )
        return test_dataset

    @property
    def test_data(self) -> TestData:
        """
        GluonTS の split によりテストデータを生成して返す。

        Returns:
        ----------
        TestData
            windows 個のウィンドウを含むテストデータ。
        """
        _, test_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
