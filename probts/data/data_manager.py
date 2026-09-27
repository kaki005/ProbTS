from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import Dataset, TrainDatasets
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository import dataset_names, datasets

from probts.data.data_utils.data_scaler import (
    IdentityScaler,
    Scaler,
    StandardScaler,
    TemporalScaler,
)
from probts.data.data_utils.data_utils import (
    df_to_mvds,
    get_rolling_test,
    split_train_val,
    truncate_test,
)
from probts.data.data_utils.get_datasets import (
    get_dataset_borders,
    get_dataset_info,
    load_dataset,
)
from probts.data.data_utils.time_features import get_lags
from probts.data.data_wrapper import ProbTSBatchData
from probts.data.datasets.gift_eval_datasets import GiftEvalDataset
from probts.data.datasets.multi_horizon_datasets import MultiHorizonDataset
from probts.data.datasets.single_horizon_datasets import (
    SingleHorizonDataset,
    TransformedIterableDataset,
)
from probts.utils.utils import ensure_list

MULTI_VARIATE_DATASETS = [
    "exchange_rate_nips",
    "solar_nips",
    "electricity_nips",
    "traffic_nips",
    "taxi_30min",
    "wiki-rolling_nips",
    "wiki2000_nips",
]


class DataManager:
    """
    データセットの読み込みと、時系列モデル向けのデータ準備を行うクラス。

    Attributes:
    ----------
    dataset : str
        データセット名 (GIFT eval の場合は読み込み時に 'gift/' と term 部分を除いた名前に更新される)。
    path : str
        データセットを格納するルートディレクトリのパス。
    history_length : int | None
        モデルへの過去入力ウィンドウの長さ (メタパラメータ設定後に確定する)。
    context_length : int | list[int] | None
        モデルへの入力コンテキスト長 (マルチホライズン時はリスト)。
    prediction_length : int | list[int] | str | None
        予測ホライズンの長さ (マルチホライズン時はリスト)。
    train_ctx_len : int | None
        学習用データセットのコンテキスト長。
    val_ctx_len : int | None
        検証用データセットのコンテキスト長。
    train_pred_len_list : list[int] | int | str | None
        学習用データセットの予測長 (長期データセットではリストに変換される)。
    val_pred_len_list : list[int] | int | str | None
        検証用データセットの予測長 (長期データセットではリストに変換される)。
    test_rolling_length : int | str
        テスト時のローリング予測のギャップウィンドウサイズ ('auto' の場合は頻度に基づき決定される)。
    split_val : bool
        学習データセットを学習用と検証用に分割するかどうか。
    scaler_type : str
        スケーラーの種類 ('none', 'standard', 'temporal')。
    context_length_factor : int
        コンテキスト長のスケーリング係数。
    timeenc : int
        時間エンコーディングの方式。
    var_specific_norm : bool
        変数ごとに独立して正規化するかどうか。
    data_path : str | None
        データセットファイルへの個別パス。
    freq : str | None
        データの頻度 (例: 'H' は毎時, 'D' は毎日)。
    multivariate : bool
        データセットが多変量かどうか。
    continuous_sample : bool
        学習時に予測ホライズンを連続的にサンプリングするかどうか。
    train_ratio : float
        学習に用いるデータの割合。
    test_ratio : float
        テストに用いるデータの割合。
    auto_search : bool
        past_len=ctx_len+pred_len とし、学習後の探索を可能にするかどうか。
    test_rolling_dict : dict[str, int]
        頻度 (小文字) から既定のローリング長への対応表。
    global_mean : torch.Tensor | None
        学習データのターゲットの変数ごとの平均。
    scaler : Scaler
        データの正規化に用いるスケーラー。
    multi_hor : bool
        マルチホライズン (複数の予測長) で扱うかどうか。
    dataset_raw : pd.DataFrame | TrainDatasets | GiftEvalDataset
        読み込んだ生データセット。
    train_iter_dataset : TransformedIterableDataset
        学習用イテラブルデータセット。
    val_iter_dataset : TransformedIterableDataset | dict[str, TransformedIterableDataset] | None
        検証用イテラブルデータセット (マルチホライズン時は予測長をキーとする辞書、検証セットが無い場合は None)。
    test_iter_dataset : TransformedIterableDataset | dict[str, TransformedIterableDataset]
        テスト用イテラブルデータセット (マルチホライズン時は予測長をキーとする辞書)。
    time_feat_dim : int
        時間特徴量の次元数。
    target_dim : int
        ターゲット変数の次元数。
    lags_list : list[int]
        頻度に基づくラグのリスト。
    train_ctx_len_list : list[int]
        学習用コンテキスト長のリスト (長期データセットのみ)。
    val_ctx_len_list : list[int]
        検証用コンテキスト長のリスト (長期データセットのみ)。
    test_ctx_len_list : list[int]
        テスト用コンテキスト長のリスト (長期データセットのみ)。
    test_pred_len_list : list[int]
        テスト用予測長のリスト (長期データセットのみ)。
    data_stamp : np.ndarray
        事前計算済みの時間特徴量 (長期データセットのみ)。
    border_begin : list[int]
        学習・検証・テスト区間の開始インデックス (長期データセットのみ)。
    border_end : list[int]
        学習・検証・テスト区間の終了インデックス (長期データセットのみ)。
    num_test_dates : int
        テスト日数 (ローリング評価のウィンドウ数, 短期データセットのみ)。
    """

    dataset: str
    path: str
    history_length: int | None
    context_length: int | list[int] | None
    prediction_length: int | list[int] | str | None
    train_ctx_len: int | None
    val_ctx_len: int | None
    train_pred_len_list: list[int] | int | str | None
    val_pred_len_list: list[int] | int | str | None
    test_rolling_length: int | str
    split_val: bool
    scaler_type: str
    context_length_factor: int
    timeenc: int
    var_specific_norm: bool
    data_path: str | None
    freq: str | None
    multivariate: bool
    continuous_sample: bool
    train_ratio: float
    test_ratio: float
    auto_search: bool
    test_rolling_dict: dict[str, int]
    global_mean: torch.Tensor | None
    scaler: Scaler
    multi_hor: bool
    dataset_raw: pd.DataFrame | TrainDatasets | GiftEvalDataset
    train_iter_dataset: TransformedIterableDataset
    val_iter_dataset: (
        TransformedIterableDataset | dict[str, TransformedIterableDataset] | None
    )
    test_iter_dataset: TransformedIterableDataset | dict[str, TransformedIterableDataset]
    time_feat_dim: int
    target_dim: int
    lags_list: list[int]
    train_ctx_len_list: list[int]
    val_ctx_len_list: list[int]
    test_ctx_len_list: list[int]
    test_pred_len_list: list[int]
    data_stamp: np.ndarray
    border_begin: list[int]
    border_end: list[int]
    num_test_dates: int

    def __init__(
        self,
        dataset: str,
        path: str = "./datasets",
        history_length: int | None = None,
        context_length: int | None = None,
        prediction_length: list[int] | int | str | None = None,
        train_ctx_len: int | None = None,
        train_pred_len_list: list[int] | int | str | None = None,
        val_ctx_len: int | None = None,
        val_pred_len_list: list[int] | int | str | None = None,
        test_rolling_length: int | str = 96,
        split_val: bool = True,
        scaler: str = "none",
        context_length_factor: int = 1,
        timeenc: int = 1,
        var_specific_norm: bool = True,
        data_path: str | None = None,
        freq: str | None = None,
        multivariate: bool = True,
        continuous_sample: bool = False,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        auto_search: bool = False,
    ) -> None:
        """
        DataManager を初期化し、データセットを読み込んで学習・検証・テスト用データセットを準備する。

        Parameters:
        ----------
        dataset : str
            読み込むデータセット名。例: "etth1", "electricity_ltsf" など。
        path : str, optional, default='./datasets'
            データセットを格納するルートディレクトリのパス。
        history_length : int | None, optional, default=None
            モデルへの過去入力ウィンドウの長さ。
            指定しない場合は `context_length` とラグ特徴量から自動計算される。
        context_length : int | None, optional, default=None
            モデルへの入力コンテキスト長。
        prediction_length : list[int] | int | str | None, optional, default=None
            モデルの予測ホライズンの長さ。以下のいずれか:
            - int: 固定の予測長。
            - list: マルチホライズン学習用の可変予測長。
            - str: 複数予測長の文字列表現。例: '96-192-336-720' は [96, 192, 336, 720] を表す。
        train_ctx_len : int | None, optional, default=None
            学習用データセットのコンテキスト長。
            指定しない場合は `context_length` の値が使われる。
        train_pred_len_list : list[int] | int | str | None, optional, default=None
            学習用データセットの予測長のリスト。
            指定しない場合は `prediction_length` の値が使われる。
        val_ctx_len : int | None, optional, default=None
            検証用データセットのコンテキスト長。
            指定しない場合は `context_length` の値が使われる。
        val_pred_len_list : list[int] | int | str | None, optional, default=None
            検証用データセットの予測長のリスト。
            指定しない場合は `prediction_length` の値が使われる。
        test_rolling_length : int | str, optional, default=96
            テスト時のローリング予測に用いるギャップウィンドウサイズ。
            - `auto` を指定した場合はデータ頻度に基づいて動的に決定される
            (例: 'H' -> 24, 'D' -> 7, 'W' -> 4)。
        split_val : bool, optional, default=True
            学習データセットを学習用と検証用に分割するかどうか。
        scaler : str, optional, default='none'
            データセットに適用する正規化・スケーリングの種類。以下のいずれか:
            - 'none': スケーリングなし。
            - 'standard': 標準化 (z-score)。
            - 'temporal': 平均スケーリングによる正規化。
        context_length_factor : int, optional, default=1
            コンテキスト長のスケーリング係数。`context_length` を動的に調整できる。
        timeenc : int, optional, default=1
            時間エンコーディングの方式。以下のいずれか:
            - 0: 時間特徴量の次元は 5 で、`month, day, weekday, hour, minute` を含む。
            - 1: 周期的な時間特徴量 (例: タイムスタンプの sine/cosine)。
            - 2: 生のタイムスタンプ情報。
        var_specific_norm : bool, optional, default=True
            変数ごとに独立して正規化するかどうか。`scaler='standard'` の場合のみ有効。
        data_path : str | None, optional, default=None
            データセットファイルへの個別パス。
        freq : str | None, optional, default=None
            データの頻度 (例: 'H' は毎時, 'D' は毎日)。
        multivariate : bool, optional, default=True
            データセットが多変量かどうか。
        continuous_sample : bool, optional, default=False
            学習時に予測ホライズンを連続的にサンプリングするかどうか。
        train_ratio : float, optional, default=0.7
            学習に用いるデータの割合。既定はデータの 70%。
        test_ratio : float, optional, default=0.2
            テストに用いるデータの割合。既定はデータの 20%。
        auto_search : bool, optional, default=False
            past_len=ctx_len+pred_len とし、学習後の探索を可能にする。
        """

        self.dataset = dataset
        self.path = path
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.train_ctx_len = (
            train_ctx_len if train_ctx_len is not None else context_length
        )
        self.val_ctx_len = val_ctx_len if val_ctx_len is not None else context_length
        self.train_pred_len_list = (
            train_pred_len_list
            if train_pred_len_list is not None
            else prediction_length
        )
        self.val_pred_len_list = (
            val_pred_len_list if val_pred_len_list is not None else prediction_length
        )
        self.test_rolling_length = test_rolling_length
        self.split_val = split_val
        self.scaler_type = scaler
        self.context_length_factor = context_length_factor
        self.timeenc = timeenc
        self.var_specific_norm = var_specific_norm
        self.data_path = data_path
        self.freq = freq
        self.multivariate = multivariate
        self.continuous_sample = continuous_sample
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.auto_search = auto_search

        self.test_rolling_dict = {"h": 24, "d": 7, "b": 5, "w": 4, "min": 60}
        self.global_mean = None

        # Configure scaler
        self.scaler = self._configure_scaler(self.scaler_type)

        # Load dataset and prepare for processing
        if dataset in dataset_names:
            self.multi_hor = False
            self._load_short_term_dataset()
        elif self.is_gift_eval:
            self.multi_hor = False
            # Load GIFT eval datasets from salesforce
            self._load_gift_eval_dataset()
        else:
            # Process context and prediction lengths
            self._process_context_and_prediction_lengths()
            self._load_long_term_dataset()
            # Print configuration details
            self._print_configurations()

    def _configure_scaler(self, scaler_type: str) -> Scaler:
        """
        スケーラーを設定する。

        Parameters:
        ----------
        scaler_type : str
            スケーラーの種類 ('standard', 'temporal', それ以外は恒等変換)。

        Returns:
        ----------
        Scaler
            生成したスケーラー。
        """
        if scaler_type == "standard":
            return StandardScaler(var_specific=self.var_specific_norm)
        elif scaler_type == "temporal":
            return TemporalScaler()
        return IdentityScaler()

    def _load_gift_eval_dataset(self) -> None:
        """
        Salesforce の GIFT eval データセットを読み込み、学習・検証・テスト用データセットを準備する。

        データセット名は 'gift/<name>/<term>' 形式を想定する。
        """
        parts = self.dataset[5:].split("/")  # Remove first 'gift/'
        self.dataset = "/".join(parts[:-1])  # Join all parts except last one with '/'
        gift_term = parts[-1]  # corresponding to "term" parameter in GiftEvalDataset
        TO_UNIVARIATE = False
        self.dataset_raw = GiftEvalDataset(
            self.dataset, term=gift_term, to_univariate=TO_UNIVARIATE
        )
        self._set_meta_parameters(
            self.dataset_raw.target_dim,
            self.dataset_raw.freq,
            self.dataset_raw.prediction_length,
        )

        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_,
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate,
        )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(
            self.dataset_raw.training_dataset, mode="train"
        )
        self.val_iter_dataset = dataset_loader.get_iter_dataset(
            self.dataset_raw.validation_dataset, mode="val"
        )
        self.test_iter_dataset = dataset_loader.get_iter_dataset(
            self.dataset_raw.test_dataset, mode="test"
        )
        self.time_feat_dim = dataset_loader.time_feat_dim
        # TODO: Implement global mean for GIFT eval datasets
        # self.global_mean = torch.mean(torch.tensor(self.dataset_raw.training_dataset[0]['target']), dim=-1)

    def _load_short_term_dataset(self) -> None:
        """
        GluonTS を用いて短期予測用データセットを読み込む。
        """
        print(f"Loading Short-term Dataset: {self.dataset}")
        self.dataset_raw = datasets.get_dataset(
            self.dataset, path=Path(self.path), regenerate=True
        )
        metadata = self.dataset_raw.metadata
        if self.is_univar_dataset:
            target_dim = 1
        else:
            target_dim = metadata.feat_static_cat[0].cardinality
        self._set_meta_parameters(
            target_dim, metadata.freq.upper(), metadata.prediction_length
        )
        self.prepare_STSF_dataset(self.dataset)

    def _set_meta_parameters(
        self, target_dim: int | str, freq: str, prediction_length: int
    ) -> None:
        """
        ベースデータセットのメタ情報からメタパラメータを設定する。

        Parameters:
        ----------
        target_dim : int | str
            ターゲット変数の次元数 (int に変換される)。
        freq : str
            データの頻度。
        prediction_length : int
            予測ホライズンの長さ。
        """
        self.target_dim = int(target_dim)
        self.multivariate = self.target_dim > 1
        self.freq = freq
        self.lags_list = get_lags(self.freq)
        self.prediction_length = prediction_length
        self.context_length = (
            self.context_length or self.prediction_length * self.context_length_factor
        )
        self.history_length = self.history_length or (
            self.context_length + max(self.lags_list)
        )

    def _process_context_and_prediction_lengths(self) -> None:
        """
        マルチホライズン処理のため、コンテキスト長と予測長をリストに変換する。

        各フェーズのコンテキスト長が単一であることを検証し、multi_hor を設定する。
        """
        self.train_ctx_len_list = ensure_list(
            self.train_ctx_len, default_value=self.context_length
        )
        self.val_ctx_len_list = ensure_list(
            self.val_ctx_len, default_value=self.context_length
        )
        self.test_ctx_len_list = ensure_list(self.context_length)
        self.train_pred_len_list = ensure_list(
            self.train_pred_len_list, default_value=self.prediction_length
        )
        self.val_pred_len_list = ensure_list(
            self.val_pred_len_list, default_value=self.prediction_length
        )
        self.test_pred_len_list = ensure_list(self.prediction_length)

        # Validate context length support
        assert len(self.train_ctx_len_list) == 1, (
            "Assign a single context length for training."
        )
        assert len(self.val_ctx_len_list) == 1, (
            "Assign a single context length for validation."
        )
        assert len(self.test_ctx_len_list) == 1, (
            "Assign a single context length for testing."
        )

        self.multi_hor = (
            len(self.train_pred_len_list) > 1
            or len(self.val_pred_len_list) > 1
            or len(self.test_pred_len_list) > 1
        )

    def _load_long_term_dataset(self) -> None:
        """
        長期予測用データセットまたはカスタムデータセットを読み込む。

        Raises:
        ----------
        ValueError
            context_length または prediction_length が指定されていない場合。
        """
        print(f"Loading Long-term Dataset: {self.dataset}")
        if not self.context_length or not self.prediction_length:
            raise ValueError("context_length or prediction_length must be specified.")

        data_path, self.freq = get_dataset_info(
            self.dataset, data_path=self.data_path, freq=self.freq
        )
        self.dataset_raw, self.data_stamp, self.target_dim, data_size = load_dataset(
            self.path,
            data_path,
            freq=self.freq,
            timeenc=self.timeenc,
            multivariate=self.multivariate,
        )
        self.border_begin, self.border_end = get_dataset_borders(
            self.dataset,
            data_size,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
        )
        self._set_meta_parameters_from_raw(data_size)
        self.prepare_dataset()

    def _set_meta_parameters_from_raw(self, data_size: int) -> None:
        """
        生データセットから直接メタパラメータを設定する。

        Parameters:
        ----------
        data_size : int
            データセットのタイムスタンプの総数。

        Raises:
        ----------
        NotImplementedError
            カスタム単変量データセットが指定された場合。
        """
        self.lags_list = get_lags(self.freq)
        self.prediction_length = (
            ensure_list(self.prediction_length)
            if self.multi_hor
            else self.prediction_length
        )
        self.context_length = (
            ensure_list(self.context_length) if self.multi_hor else self.context_length
        )
        self.history_length = self.history_length or (
            max(self.context_length) + max(self.lags_list)
            if self.multi_hor
            else self.context_length + max(self.lags_list)
        )
        if not self.multivariate:
            self.target_dim = 1
            raise NotImplementedError(
                "Customized univariate datasets are not yet supported."
            )
        assert data_size >= self.border_end[2], "border_end index exceeds dataset size!"

        # define the test_rolling_length
        if self.test_rolling_length == "auto":
            if self.freq.lower() in self.test_rolling_dict:
                self.test_rolling_length = self.test_rolling_dict[self.freq.lower()]
            else:
                self.test_rolling_length = 24

    def prepare_dataset(self) -> None:
        """
        学習・検証・テスト用データセットを準備する。
        """
        # Split raw data into train, validation, and test sets
        train_data = self.dataset_raw[: self.border_end[0]]
        val_data = self.dataset_raw[: self.border_end[1]]
        test_data = self.dataset_raw[: self.border_end[2]]

        # Calculate statictics using training data
        self.scaler.fit(torch.tensor(train_data.values))

        # Convert dataframes to multivariate datasets
        train_set = df_to_mvds(train_data, freq=self.freq)
        val_set = df_to_mvds(val_data, freq=self.freq)
        test_set = df_to_mvds(test_data, freq=self.freq)

        train_grouper = MultivariateGrouper(max_target_dim=self.target_dim)
        test_grouper = MultivariateGrouper(max_target_dim=self.target_dim)

        group_train_set = train_grouper(train_set)
        group_val_set = test_grouper(val_set)
        group_test_set = test_grouper(test_set)

        if self.multi_hor:
            # Handle multi-horizon datasets
            dataset_loader = self._prepare_multi_horizon_datasets(
                group_val_set, group_test_set
            )
        else:
            # Handle single-horizon datasets
            dataset_loader = self._prepare_single_horizon_datasets(
                group_val_set, group_test_set
            )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(
            group_train_set,
            mode="train",
            data_stamp=self.data_stamp[: self.border_end[0]],
        )

        self.time_feat_dim = dataset_loader.time_feat_dim
        self.global_mean = torch.mean(
            torch.tensor(group_train_set[0]["target"]), dim=-1
        )

    def _prepare_multi_horizon_datasets(
        self, group_val_set: Dataset, group_test_set: Dataset
    ) -> MultiHorizonDataset:
        """
        検証・テスト用のマルチホライズンデータセットを準備する。

        Parameters:
        ----------
        group_val_set : Dataset
            多変量にグループ化された検証用データセット。
        group_test_set : Dataset
            多変量にグループ化されたテスト用データセット。

        Returns:
        ----------
        MultiHorizonDataset
            学習用データセットの生成にも用いるデータセットローダ。
        """
        self.val_iter_dataset = {}
        self.test_iter_dataset = {}
        dataset_loader = MultiHorizonDataset(
            input_names=ProbTSBatchData.input_names_,
            freq=self.freq,
            train_ctx_range=self.train_ctx_len_list,
            train_pred_range=self.train_pred_len_list,
            val_ctx_range=self.val_ctx_len_list,
            val_pred_range=self.val_pred_len_list,
            test_ctx_range=self.test_ctx_len_list,
            test_pred_range=self.test_pred_len_list,
            multivariate=self.multivariate,
            continuous_sample=self.continuous_sample,
        )

        # Prepare validation datasets
        for pred_len in self.val_pred_len_list:
            local_group_val_set = get_rolling_test(
                "val",
                group_val_set,
                self.border_begin[1],
                self.border_end[1],
                rolling_length=self.test_rolling_length,
                pred_len=pred_len,
                freq=self.freq,
            )
            self.val_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                local_group_val_set,
                mode="val",
                data_stamp=self.data_stamp[: self.border_end[1]],
                pred_len=[pred_len],
            )

        # Prepare testing datasets
        for pred_len in self.test_pred_len_list:
            local_group_test_set = get_rolling_test(
                "test",
                group_test_set,
                self.border_begin[2],
                self.border_end[2],
                rolling_length=self.test_rolling_length,
                pred_len=pred_len,
                freq=self.freq,
            )
            self.test_iter_dataset[str(pred_len)] = dataset_loader.get_iter_dataset(
                local_group_test_set,
                mode="test",
                data_stamp=self.data_stamp[: self.border_end[2]],
                pred_len=[pred_len],
                auto_search=self.auto_search,
            )

        return dataset_loader

    def _prepare_single_horizon_datasets(
        self, group_val_set: Dataset, group_test_set: Dataset
    ) -> SingleHorizonDataset:
        """
        検証・テスト用の単一ホライズンデータセットを準備する。

        Parameters:
        ----------
        group_val_set : Dataset
            多変量にグループ化された検証用データセット。
        group_test_set : Dataset
            多変量にグループ化されたテスト用データセット。

        Returns:
        ----------
        SingleHorizonDataset
            学習用データセットの生成にも用いるデータセットローダ。
        """
        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_,
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate,
        )

        # Validation dataset
        local_group_val_set = get_rolling_test(
            "val",
            group_val_set,
            self.border_begin[1],
            self.border_end[1],
            rolling_length=self.test_rolling_length,
            pred_len=self.val_pred_len_list[0],
            freq=self.freq,
        )
        self.val_iter_dataset = dataset_loader.get_iter_dataset(
            local_group_val_set,
            mode="val",
            data_stamp=self.data_stamp[: self.border_end[1]],
        )

        # Testing dataset
        local_group_test_set = get_rolling_test(
            "test",
            group_test_set,
            self.border_begin[2],
            self.border_end[2],
            rolling_length=self.test_rolling_length,
            pred_len=self.prediction_length,
            freq=self.freq,
        )
        self.test_iter_dataset = dataset_loader.get_iter_dataset(
            local_group_test_set,
            mode="test",
            data_stamp=self.data_stamp[: self.border_end[2]],
            auto_search=self.auto_search,
        )

        return dataset_loader

    def prepare_STSF_dataset(self, dataset: str) -> None:
        """
        短期時系列予測 (STSF) 用のデータセットを準備する。

        Parameters:
        ----------
        dataset : str
            データセット名。
        """
        if dataset in MULTI_VARIATE_DATASETS:
            self.num_test_dates = int(
                len(self.dataset_raw.test) / len(self.dataset_raw.train)
            )

            train_grouper = MultivariateGrouper(max_target_dim=int(self.target_dim))
            test_grouper = MultivariateGrouper(
                num_test_dates=self.num_test_dates, max_target_dim=int(self.target_dim)
            )
            train_set = train_grouper(self.dataset_raw.train)
            test_set = test_grouper(self.dataset_raw.test)
            self.scaler.fit(torch.tensor(train_set[0]["target"].transpose(1, 0)))
            self.global_mean = torch.mean(torch.tensor(train_set[0]["target"]), dim=-1)

            # split_val
            if self.split_val:
                train_set, val_set = split_train_val(
                    train_set,
                    self.num_test_dates,
                    self.context_length,
                    self.prediction_length,
                    self.freq,
                )
            else:
                val_set = None
        else:
            self.target_dim = 1
            self.multivariate = False
            self.num_test_dates = 1
            train_set = self.dataset_raw.train
            test_set = self.dataset_raw.test
            test_set = truncate_test(
                test_set, self.context_length, self.prediction_length, self.freq
            )
            # for univariate dataset, e.g., M4 and M5, no validation set is used
            val_set = None

        if val_set is None:
            print("No validation set is used.")

        dataset_loader = SingleHorizonDataset(
            ProbTSBatchData.input_names_,
            self.history_length,
            self.context_length,
            self.prediction_length,
            self.freq,
            self.multivariate,
        )

        self.train_iter_dataset = dataset_loader.get_iter_dataset(
            train_set, mode="train"
        )
        if val_set is not None:
            self.val_iter_dataset = dataset_loader.get_iter_dataset(val_set, mode="val")
        else:
            self.val_iter_dataset = None
        self.test_iter_dataset = dataset_loader.get_iter_dataset(test_set, mode="test")
        self.time_feat_dim = dataset_loader.time_feat_dim

    def _print_configurations(self) -> None:
        """
        データセットと設定の詳細を表示する。
        """
        print(
            f"Test context length: {self.test_ctx_len_list}, prediction length: {self.test_pred_len_list}"
        )
        print(
            f"Validation context length: {self.val_ctx_len_list}, prediction length: {self.val_pred_len_list}"
        )
        print(
            f"Training context length: {self.train_ctx_len_list}, prediction lengths: {self.train_pred_len_list}"
        )
        print(f"Test rolling length: {self.test_rolling_length}")
        if self.scaler_type == "standard":
            print(f"Variable-specific normalization: {self.var_specific_norm}")

    @cached_property
    def is_gift_eval(self) -> bool:
        """
        データセットが GIFT eval データセット ('gift/' で始まる名前) かどうかを返す。

        Returns:
        ----------
        bool
            GIFT eval データセットであれば True。
        """
        return self.dataset[:5] == "gift/"

    @cached_property
    def is_univar_dataset(self) -> bool:
        """
        データセットが単変量データセット (名前に 'm4' または 'm5' を含む) かどうかを返す。

        Returns:
        ----------
        bool
            単変量データセットであれば True。
        """
        if "m4" in self.dataset or "m5" in self.dataset:
            return True
        return False
