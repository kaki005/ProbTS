# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from collections.abc import Iterator
from typing import Any

import numpy as np
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
from gluonts.itertools import Cyclic
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    SelectFields,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    Transformation,
    TransformedDataset,
    ValidationSplitSampler,
    VstackFeatures,
)
from torch.utils.data import IterableDataset

from probts.data.data_utils.time_features import (
    AddCustomizedTimeFeatures,
    fourier_time_features_from_frequency,
)


class SingleHorizonDataset:
    """
    単一ホライズン予測タスク向けに、データセットの変換とインスタンス分割を行うクラス。

    Attributes:
    ----------
    input_names_ : list[str]
        モデルが必要とする入力フィールド名のリスト。
    history_length : int
        入力として用いる過去系列ウィンドウの長さ。
    context_length : int
        コンテキスト長 (auto_search 時の過去長の計算に使用)。
    prediction_length : int
        予測ホライズンの長さ。
    freq : str
        データの頻度 (例: 'H' は毎時, 'D' は毎日)。
    expected_ndim : int
        ターゲット配列の期待次元数 (多変量なら 2, 単変量なら 1)。
    time_feat_dim : int
        時間特徴量の次元数 (create_transformation 呼び出し後に設定)。
    train_sampler : InstanceSampler
        学習用インスタンスサンプラー (get_sampler 呼び出し後に設定)。
    val_sampler : InstanceSampler
        検証用インスタンスサンプラー (get_sampler 呼び出し後に設定)。
    test_sampler : InstanceSampler
        テスト用インスタンスサンプラー (get_sampler 呼び出し後に設定)。
    """

    input_names_: list[str]
    history_length: int
    context_length: int
    prediction_length: int
    freq: str
    expected_ndim: int
    time_feat_dim: int
    train_sampler: InstanceSampler
    val_sampler: InstanceSampler
    test_sampler: InstanceSampler

    def __init__(
        self,
        input_names: list[str],
        history_length: int,
        context_length: int,
        prediction_length: int,
        freq: str,
        multivariate: bool = True,
    ) -> None:
        """
        SingleHorizonDataset を初期化する。

        Parameters:
        ----------
        input_names : list[str]
            モデルが必要とする入力フィールド名のリスト。
        history_length : int
            入力として用いる過去系列ウィンドウの長さ。
        context_length : int
            コンテキスト長。
        prediction_length : int
            予測ホライズンの長さ。
        freq : str
            データの頻度 (例: 'H' は毎時, 'D' は毎日)。
        multivariate : bool, optional, default=True
            データセットが複数のターゲット変数を含むかどうか。
        """
        super().__init__()
        self.input_names_ = input_names
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        if multivariate:
            self.expected_ndim = 2
        else:
            self.expected_ndim = 1

    def get_sampler(self) -> None:
        """
        学習・検証・テスト用のインスタンスサンプラーを生成し、属性に設定する。

        - 学習: ランダムな時点でインスタンスを生成する。
        - 検証・テスト: 常に系列の最後の時点を選択する。
        """
        # returns a set of indices at which training instances will be generated
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

        self.test_sampler = ValidationSplitSampler(
            min_past=self.history_length,
            min_future=self.prediction_length,
        )

    def create_transformation(
        self, data_stamp: np.ndarray | None = None
    ) -> Transformation:
        """
        モデル入力を準備するためのデータ変換パイプラインを生成する。
        時間特徴量や観測値インジケータなどの特徴量を付与する。

        Parameters:
        ----------
        data_stamp : np.ndarray | None, optional
            事前計算済みの時間特徴量。None の場合はデータ頻度に基づいて特徴量を生成する。

        Returns:
        ----------
        Transformation
            データセットに適用する変換の連鎖 (Chain)。
        """
        if data_stamp is None:
            if self.freq in ["M", "W", "D", "B", "H", "min", "T"]:
                time_features = fourier_time_features_from_frequency(self.freq)
            else:
                time_features = fourier_time_features_from_frequency("D")
            self.time_feat_dim = len(time_features) * 2
            time_feature_func = AddTimeFeatures
        else:
            self.time_feat_dim = data_stamp.shape[-1]
            time_features = data_stamp
            time_feature_func = AddCustomizedTimeFeatures

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=self.expected_ndim,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                time_feature_func(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(
        self, mode: str, auto_search: bool = False
    ) -> Transformation:
        """
        学習・検証・テスト用のインスタンススプリッターを生成する。

        Parameters:
        ----------
        mode : str
            動作モード。['train', 'val', 'test'] のいずれか。
        auto_search : bool, optional, default=False
            True の場合、過去長を context_length + prediction_length とする。
            False の場合は history_length を用いる。

        Returns:
        ----------
        Transformation
            学習・評価用に入力データを切り出すスプリッター変換 (InstanceSplitter + RenameFields)。
        """
        assert mode in ["train", "val", "test"]

        self.get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        if auto_search:
            past_length = self.context_length + self.prediction_length
        else:
            past_length = self.history_length

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def get_iter_dataset(
        self,
        dataset: Dataset,
        mode: str,
        data_stamp: np.ndarray | None = None,
        auto_search: bool = False,
    ) -> "TransformedIterableDataset":
        """
        学習・検証・テスト用のイテラブルデータセットを生成する。

        Parameters:
        ----------
        dataset : Dataset
            変換対象の入力データセット。
        mode : str
            動作モード。['train', 'val', 'test'] のいずれか。
        data_stamp : np.ndarray | None, optional
            事前計算済みの時間特徴量。
        auto_search : bool, optional, default=False
            検証・テスト時のスプリッターで auto_search を有効にするかどうか。

        Returns:
        ----------
        TransformedIterableDataset
            変換とインスタンス分割を適用したデータセット。
        """
        assert mode in ["train", "val", "test"]

        transform = self.create_transformation(data_stamp)
        if mode == "train":
            with env._let(max_idle_transforms=100):
                instance_splitter = self.create_instance_splitter(mode)
        else:
            instance_splitter = self.create_instance_splitter(
                mode, auto_search=auto_search
            )

        input_names = self.input_names_

        iter_dataset = TransformedIterableDataset(
            dataset,
            transform=transform + instance_splitter + SelectFields(input_names),
            is_train=True if mode == "train" else False,
        )

        return iter_dataset


class TransformedIterableDataset(IterableDataset):
    """
    変換パイプラインを逐次 (on-the-fly) 適用するイテラブルデータセット。

    Attributes:
    ----------
    transformed_dataset : TransformedDataset
        変換を適用したデータセット。学習時は元データを循環 (Cyclic) させる。
    """

    transformed_dataset: TransformedDataset

    def __init__(
        self, dataset: Dataset, transform: Transformation, is_train: bool = True
    ) -> None:
        """
        TransformedIterableDataset を初期化する。

        Parameters:
        ----------
        dataset : Dataset
            変換対象の元データセット。
        transform : Transformation
            適用する変換パイプライン。
        is_train : bool, optional, default=True
            学習用として使用するかどうか。True の場合はデータセットを循環させる。
        """
        super().__init__()

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if is_train else dataset,
            transform,
            is_train=is_train,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        変換済みデータのイテレータを返す。

        Returns:
        ----------
        Iterator[dict[str, Any]]
            変換済みの各インスタンス (フィールド名 -> 値の辞書) を返すイテレータ。
        """
        return iter(self.transformed_dataset)
