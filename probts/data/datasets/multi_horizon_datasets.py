# ---------------------------------------------------------------------------------
# Portions of this file are derived from GluonTS
# - Source: https://github.com/awslabs/gluonts
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import random
from collections.abc import Iterator

import numpy as np
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpandDimArray,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    RenameFields,
    SelectFields,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform._base import FlatMapTransformation
from gluonts.zebras._util import pad_axis
from torch.utils.data import IterableDataset

from probts.data.data_utils.time_features import (
    AddCustomizedTimeFeatures,
    fourier_time_features_from_frequency,
)
from probts.data.datasets.single_horizon_datasets import TransformedIterableDataset


class MultiHorizonDataset:
    """
    コンテキスト長と予測長を柔軟に扱い、マルチホライズン予測をサポートするデータセットクラス。

    Attributes:
    ----------
    input_names_ : list[str]
        モデルが必要とする入力フィールド名のリスト。
    train_ctx_range : int | list[int]
        学習データセットのコンテキスト長の範囲。
    train_pred_range : int | list[int]
        学習データセットの予測長の範囲。
    val_ctx_range : int | list[int]
        検証データセットのコンテキスト長の範囲。
    val_pred_range : int | list[int]
        検証データセットの予測長の範囲。
    test_ctx_range : int | list[int]
        テストデータセットのコンテキスト長の範囲。
    test_pred_range : int | list[int]
        テストデータセットの予測長の範囲。
    continuous_sample : bool
        train_pred_range から予測ホライズンを連続的にサンプリングするかどうか。
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
    train_ctx_range: int | list[int]
    train_pred_range: int | list[int]
    val_ctx_range: int | list[int]
    val_pred_range: int | list[int]
    test_ctx_range: int | list[int]
    test_pred_range: int | list[int]
    continuous_sample: bool
    freq: str
    expected_ndim: int
    time_feat_dim: int
    train_sampler: InstanceSampler
    val_sampler: InstanceSampler
    test_sampler: InstanceSampler

    def __init__(
        self,
        input_names: list[str],
        freq: str,
        train_ctx_range: int | list[int],
        train_pred_range: int | list[int],
        val_ctx_range: int | list[int],
        val_pred_range: int | list[int],
        test_ctx_range: int | list[int],
        test_pred_range: int | list[int],
        multivariate: bool = True,
        continuous_sample: bool = False,
    ) -> None:
        """
        MultiHorizonDataset を初期化する。

        Parameters:
        ----------
        input_names : list[str]
            モデルが必要とする入力フィールド名のリスト。
        freq : str
            データの頻度 (例: 'H' は毎時, 'D' は毎日)。
        train_ctx_range : int | list[int]
            学習データセットのコンテキスト長の範囲。
        train_pred_range : int | list[int]
            学習データセットの予測長の範囲。
        val_ctx_range : int | list[int]
            検証データセットのコンテキスト長の範囲。
        val_pred_range : int | list[int]
            検証データセットの予測長の範囲。
        test_ctx_range : int | list[int]
            テストデータセットのコンテキスト長の範囲。
        test_pred_range : int | list[int]
            テストデータセットの予測長の範囲。
        multivariate : bool, optional, default=True
            データセットが複数のターゲット変数を含むかどうか。
        continuous_sample : bool, optional, default=False
            train_pred_range から予測ホライズンを連続的にサンプリングするかどうか。
        """
        super().__init__()
        self.input_names_ = input_names
        self.train_ctx_range = train_ctx_range
        self.train_pred_range = train_pred_range
        self.val_ctx_range = val_ctx_range
        self.val_pred_range = val_pred_range
        self.test_ctx_range = test_ctx_range
        self.test_pred_range = test_pred_range
        self.continuous_sample = continuous_sample

        self.freq = freq
        if multivariate:
            self.expected_ndim = 2
        else:
            self.expected_ndim = 1

    def get_sampler(self) -> None:
        """
        学習・検証・テスト用のインスタンスサンプラーを生成し、属性に設定する。
        サンプラーは各モードでデータインスタンスをどのように選択するかを制御する。

        - 学習: コンテキスト長・予測長の最小値を用いてランダムな時点を選択する。
        - 検証・テスト: 最大値を用いて系列の最後の時点を選択する。
        """

        # for training
        train_min_past = min(self.train_ctx_range)
        train_min_future = min(self.train_pred_range)

        # for validation
        val_min_past = max(self.val_ctx_range)
        val_min_future = max(self.val_pred_range)

        # for testing
        if type(self.test_ctx_range).__name__ == "list":
            test_min_past = max(self.test_ctx_range)
        else:
            test_min_past = self.test_ctx_range

        if type(self.test_pred_range).__name__ == "list":
            test_min_future = max(self.test_pred_range)
        else:
            test_min_future = self.test_pred_range

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=train_min_past,
            min_future=train_min_future,
        )

        self.val_sampler = ValidationSplitSampler(
            min_past=val_min_past,
            min_future=val_min_future,
        )

        self.test_sampler = ValidationSplitSampler(
            min_past=test_min_past,
            min_future=test_min_future,
        )

    def create_transformation(
        self,
        data_stamp: np.ndarray | None = None,
        pred_len: list[int] | None = None,
    ) -> Transformation:
        """
        データ前処理のための変換パイプラインを生成する。
        時間特徴量や観測値インジケータなどの特徴量を付与する。

        Parameters:
        ----------
        data_stamp : np.ndarray | None, optional
            事前計算済みの時間特徴量。None の場合はデータ頻度に基づいて特徴量を生成する。
        pred_len : list[int] | None, optional
            変換に用いる予測長のリスト (最大値が使用される)。
            None の場合は学習用予測長範囲の最大値を用いる。

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

        if pred_len is None:
            pred_len = max(self.train_pred_range)
        else:
            pred_len = max(pred_len)

        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=self.expected_ndim,
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
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
                    pred_length=pred_len,
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
        self,
        mode: str,
        pred_len: list[int] | None = None,
        auto_search: bool = False,
    ) -> Transformation:
        """
        時系列を切り出すためのインスタンススプリッターを生成する。

        Parameters:
        ----------
        mode : str
            データセットのモード。['train', 'val', 'test'] のいずれか。
        pred_len : list[int] | None, optional
            検証・テスト時の予測長。None の場合は事前定義された範囲を用いる。
        auto_search : bool, optional, default=False
            テスト時に True の場合、過去長を
            [max(test_ctx_range) + max(future_length)] とする。

        Returns:
        ----------
        Transformation
            時系列を切り出す変換 (MultiHorizonSplitter + RenameFields)。
        """
        assert mode in ["train", "val", "test"]

        self.get_sampler()
        instance_sampler = {
            "train": self.train_sampler,
            "val": self.val_sampler,
            "test": self.test_sampler,
        }[mode]

        if mode == "train":
            past_length = self.train_ctx_range
            future_length = self.train_pred_range
        elif mode == "val":
            past_length = self.val_ctx_range
            if pred_len is None:
                future_length = self.val_pred_range
            else:
                future_length = pred_len
        else:
            if pred_len is None:
                future_length = self.test_pred_range
            else:
                future_length = pred_len

            if auto_search:
                past_length = [max(self.test_ctx_range) + max(future_length)]
            else:
                past_length = self.test_ctx_range

        return MultiHorizonSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=future_length,
            mode=mode,
            continuous_sample=self.continuous_sample,
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
        pred_len: list[int] | None = None,
        auto_search: bool = False,
    ) -> IterableDataset:
        """
        変換とスプリッターを適用したイテラブルデータセットを生成する。

        Parameters:
        ----------
        dataset : Dataset
            変換対象の入力データセット。
        mode : str
            動作モード。['train', 'val', 'test'] のいずれか。
        data_stamp : np.ndarray | None, optional
            事前計算済みの時間特徴量。
        pred_len : list[int] | None, optional
            検証・テスト時の予測長。
        auto_search : bool, optional, default=False
            検証・テスト時のスプリッターで auto_search を有効にするかどうか。

        Returns:
        ----------
        IterableDataset
            学習・評価に利用可能な変換済みデータセット (TransformedIterableDataset)。
        """
        assert mode in ["train", "val", "test"]

        transform = self.create_transformation(data_stamp, pred_len=pred_len)

        if mode == "train":
            with env._let(max_idle_transforms=100):
                instance_splitter = self.create_instance_splitter(mode)
        else:
            instance_splitter = self.create_instance_splitter(
                mode, pred_len=pred_len, auto_search=auto_search
            )

        input_names = self.input_names_

        iter_dataset = TransformedIterableDataset(
            dataset,
            transform=transform + instance_splitter + SelectFields(input_names),
            is_train=True if mode == "train" else False,
        )

        return iter_dataset


class MultiHorizonSplitter(FlatMapTransformation):
    """
    指定したサンプラーで選択された時点でターゲットおよびその他の時系列フィールドを
    切り出し、データセットからインスタンスを分割する変換クラス。
    全ての時系列フィールドは同じ時点から開始することを前提とする。

    時間軸は常に最後の軸であることを前提とする。

    ``target_field`` および ``time_series_fields`` の各フィールドは削除され、
    それぞれ `past_` と `future_` を接頭辞に持つ 2 つの新しいフィールドに置き換えられる。

    また、各時点の値がパディングかどうかを示す ``past_is_pad`` も追加される。

    Attributes:
    ----------
    instance_sampler : InstanceSampler
        時系列からサンプリング位置のインデックスを与えるインスタンスサンプラー。
    past_length : int | list[int]
        予測前に参照するターゲットの長さ (またはその候補リスト)。
    future_length : int | list[int]
        予測すべきターゲットの長さ (またはその候補リスト)。
    continuous_sample : bool
        学習時に past_length / future_length の範囲から連続的に長さをサンプリングするかどうか。
    lead_time : int
        過去ウィンドウと未来ウィンドウの間のギャップ。
    output_NTC : bool
        時系列出力を (time, dimension) レイアウトにするか (True)、
        (dimension, time) レイアウトにするか (False)。
    ts_fields : list[str]
        ターゲットと同じ区間で分割される時系列フィールド名のリスト。
    target_field : str
        ターゲットを含むフィールド名。
    is_pad_field : str
        パディングの有無を示す出力フィールド名。
    start_field : str
        時系列の開始日時を含むフィールド名。
    forecast_start_field : str
        予測開始時点を格納する出力フィールド名。
    dummy_value : float
        パディングに用いる値。
    mode : str
        動作モード。['train', 'val', 'test'] のいずれか。
    """

    instance_sampler: InstanceSampler
    past_length: int | list[int]
    future_length: int | list[int]
    continuous_sample: bool
    lead_time: int
    output_NTC: bool
    ts_fields: list[str]
    target_field: str
    is_pad_field: str
    start_field: str
    forecast_start_field: str
    dummy_value: float
    mode: str

    # @validated()
    def __init__(
        self,
        target_field: str,
        is_pad_field: str,
        start_field: str,
        forecast_start_field: str,
        instance_sampler: InstanceSampler,
        past_length: int | list[int],
        future_length: int | list[int],
        mode: str,
        lead_time: int = 0,
        output_NTC: bool = True,
        time_series_fields: list[str] = [],
        dummy_value: float = 0.0,
        continuous_sample: bool = False,
    ) -> None:
        """
        MultiHorizonSplitter を初期化する。

        Parameters:
        ----------
        target_field : str
            ターゲットを含むフィールド名。
        is_pad_field : str
            パディングの有無を示す出力フィールド名。
        start_field : str
            時系列の開始日時を含むフィールド名。
        forecast_start_field : str
            予測開始時点を格納する出力フィールド名。
        instance_sampler : InstanceSampler
            時系列からサンプリング位置のインデックスを与えるインスタンスサンプラー。
        past_length : int | list[int]
            予測前に参照するターゲットの長さ (またはその候補リスト)。
        future_length : int | list[int]
            予測すべきターゲットの長さ (またはその候補リスト)。
        mode : str
            動作モード。['train', 'val', 'test'] のいずれか。
        lead_time : int, optional, default=0
            過去ウィンドウと未来ウィンドウの間のギャップ。
        output_NTC : bool, optional, default=True
            時系列出力を (time, dimension) レイアウトにするか (True)、
            (dimension, time) レイアウトにするか (False)。
        time_series_fields : list[str], optional, default=[]
            ターゲットと同じ区間で分割される時系列フィールド名のリスト。
        dummy_value : float, optional, default=0.0
            パディングに用いる値。
        continuous_sample : bool, optional, default=False
            学習時に長さの範囲から連続的にサンプリングするかどうか。
        """
        super().__init__()

        # assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.continuous_sample = continuous_sample

        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value
        self.mode = mode

    def _past(self, col_name: str) -> str:
        """
        過去部分のフィールド名を生成する。

        Parameters:
        ----------
        col_name : str
            元のフィールド名。

        Returns:
        ----------
        str
            `past_` を接頭辞に付与したフィールド名。
        """
        return f"past_{col_name}"

    def _future(self, col_name: str) -> str:
        """
        未来部分のフィールド名を生成する。

        Parameters:
        ----------
        col_name : str
            元のフィールド名。

        Returns:
        ----------
        str
            `future_` を接頭辞に付与したフィールド名。
        """
        return f"future_{col_name}"

    def _split_array(
        self, array: np.ndarray, idx: int, past_length: int, future_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        配列を指定位置で過去部分と未来部分に分割する。
        過去部分が不足する場合は dummy_value で左側をパディングする。

        Parameters:
        ----------
        array : np.ndarray
            分割対象の配列 (時間軸は最後の軸)。
        idx : int
            分割位置 (予測開始位置) のインデックス。
        past_length : int
            切り出す過去部分の長さ。
        future_length : int
            切り出す未来部分の長さ。

        Returns:
        ----------
        tuple[np.ndarray, np.ndarray]
            (過去部分, 未来部分) の配列のタプル。
        """
        if idx >= past_length:
            past_piece = array[..., idx - past_length : idx]
        else:
            past_piece = pad_axis(
                array[..., :idx],
                axis=-1,
                left=past_length - idx,
                value=self.dummy_value,
            )

        future_start = idx + self.lead_time
        future_slice = slice(future_start, future_start + future_length)
        future_piece = array[..., future_slice]

        return past_piece, future_piece

    def _split_instance(self, entry: DataEntry, idx: int, is_train: bool) -> DataEntry:
        """
        1 つのデータエントリを指定位置で過去・未来に分割したインスタンスを生成する。

        Parameters:
        ----------
        entry : DataEntry
            分割対象のデータエントリ。
        idx : int
            分割位置 (予測開始位置) のインデックス。
        is_train : bool
            学習時かどうか。True の場合は過去長・予測長をランダムにサンプリングし、
            False の場合は最大値を用いる。

        Returns:
        ----------
        DataEntry
            past_/future_ フィールド、パディング指標、予測開始時点、
            context_length および prediction_length を付与した新しいエントリ。
        """
        slice_cols = self.ts_fields + [self.target_field]
        dtype = entry[self.target_field].dtype
        entry = entry.copy()

        if is_train:
            if self.continuous_sample:
                past_len = random.randint(min(self.past_length), max(self.past_length))
                pred_len = random.randint(
                    min(self.future_length), max(self.future_length)
                )
            else:
                past_len = random.choice(self.past_length)
                pred_len = random.choice(self.future_length)
        else:
            past_len = max(self.past_length)
            pred_len = max(self.future_length)

        for ts_field in slice_cols:
            past_piece, future_piece = self._split_array(
                entry[ts_field], idx, past_length=past_len, future_length=pred_len
            )

            if self.output_NTC:
                past_piece = past_piece.transpose()
                future_piece = future_piece.transpose()

            entry[self._past(ts_field)] = past_piece
            entry[self._future(ts_field)] = future_piece
            del entry[ts_field]

        pad_indicator = np.zeros(past_len, dtype=dtype)
        pad_length = max(past_len - idx, 0)
        pad_indicator[:pad_length] = 1

        entry[self._past(self.is_pad_field)] = pad_indicator
        entry[self.forecast_start_field] = (
            entry[self.start_field] + idx + self.lead_time
        )
        entry["context_length"] = past_len
        entry["prediction_length"] = pred_len

        return entry

    def flatmap_transform(
        self, entry: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        """
        インスタンスサンプラーで選択された各位置についてエントリを分割し、順に返す。

        Parameters:
        ----------
        entry : DataEntry
            分割対象のデータエントリ。
        is_train : bool
            学習時かどうか。

        Returns:
        ----------
        Iterator[DataEntry]
            分割済みインスタンスを返すイテレータ。
        """
        sampled_indices = self.instance_sampler(entry[self.target_field])

        for idx in sampled_indices:
            yield self._split_instance(entry, idx, is_train)
