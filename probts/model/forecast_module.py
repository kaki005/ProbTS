from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from torch import optim

from probts.data import ProbTSBatchData
from probts.data.data_utils.data_scaler import Scaler
from probts.model.forecaster import Forecaster
from probts.utils.evaluator import Evaluator
from probts.utils.metrics import *
from probts.utils.save_utils import (
    calculate_weighted_average,
    get_hor_str,
    load_checkpoint,
    update_metrics,
)
from probts.utils.utils import init_class_helper


def get_weights(sampling_weight_scheme: str, max_hor: int) -> torch.Tensor | None:
    """
    ホライズンごとの損失重みを生成する。

    Parameters:
    ----------
    sampling_weight_scheme : str
        重み付け方式。['random', 'const', 'none'] のいずれか。
        'random' は対数的に減衰する重み、'const' は一様重み、'none' は重みなし。
    max_hor : int
        最大ホライズン長 (重みベクトルの長さ)。

    Returns:
    ----------
    torch.Tensor | None
        形状 [max_hor] の重みテンソル。'none' の場合は None。
    """
    if sampling_weight_scheme == "random":
        i_array = np.linspace(1 + 1e-5, max_hor - 1e-3, max_hor)
        w = (1 / max_hor) * (np.log(max_hor) - np.log(i_array))
    elif sampling_weight_scheme == "const":
        w = np.array([1 / max_hor] * max_hor)
    elif sampling_weight_scheme == "none":
        return None
    else:
        raise ValueError(f"Invalid sampling scheme {sampling_weight_scheme}.")

    return torch.tensor(w)


class ProbTSForecastModule(pl.LightningModule):
    """
    Forecaster をラップし、学習・検証・テスト・予測のループを提供する LightningModule。

    Attributes:
    ----------
    num_samples : int
        確率的予測で生成するサンプル数。
    learning_rate : float
        学習率 (optimizer_config が None の場合に Adam で使用)。
    load_from_ckpt : str | None
        読み込むチェックポイントのパス。
    train_pred_len_list : list[int] | None
        学習時の予測長のリスト。
    forecaster : Forecaster
        予測モデル本体。
    optimizer_config : dict[str, Any] | None
        オプティマイザの設定 ("class_name" と "init_args" を持つ辞書)。
    scheduler_config : dict[str, Any] | None
        学習率スケジューラの設定 ("class_name" と "init_args" を持つ辞書)。
    scaler : Scaler | None
        データの正規化に用いるスケーラー。
    evaluator : Evaluator
        予測結果の評価器。
    sampling_weight_scheme : str
        ホライズンごとの損失重み付け方式。
    metrics_dict : dict[str, Any]
        エポック中に蓄積される評価指標 (エポック開始時に初期化)。
    hor_metrics : dict[str, dict[str, Any]]
        ホライズンごとに蓄積される評価指標 (エポック開始時に初期化)。
    batch_size : list[int]
        各バッチのサイズの履歴 (加重平均の計算に使用)。
    avg_metrics : dict[str, Any]
        テストエポック終了時に計算される平均評価指標。
    avg_hor_metrics : dict[str, dict[str, Any]]
        テストエポック終了時に計算されるホライズンごとの平均評価指標。
    """

    num_samples: int
    learning_rate: float
    load_from_ckpt: str | None
    train_pred_len_list: list[int] | None
    forecaster: Forecaster
    optimizer_config: dict[str, Any] | None
    scheduler_config: dict[str, Any] | None
    scaler: Scaler | None
    evaluator: Evaluator
    sampling_weight_scheme: str
    metrics_dict: dict[str, Any]
    hor_metrics: dict[str, dict[str, Any]]
    batch_size: list[int]
    avg_metrics: dict[str, Any]
    avg_hor_metrics: dict[str, dict[str, Any]]

    def __init__(
        self,
        forecaster: Forecaster,
        scaler: Scaler | None = None,
        train_pred_len_list: list[int] | None = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        quantiles_num: int = 10,
        load_from_ckpt: str | None = None,
        sampling_weight_scheme: str = "none",
        optimizer_config: dict[str, Any] | None = None,
        lr_scheduler_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        ProbTSForecastModule を初期化する。

        Parameters:
        ----------
        forecaster : Forecaster
            予測モデル本体。
        scaler : Scaler | None, optional, default=None
            データの正規化に用いるスケーラー。
        train_pred_len_list : list[int] | None, optional, default=None
            学習時の予測長のリスト。
        num_samples : int, optional, default=100
            確率的予測で生成するサンプル数。
        learning_rate : float, optional, default=1e-3
            学習率。
        quantiles_num : int, optional, default=10
            評価に用いる分位点の数。
        load_from_ckpt : str | None, optional, default=None
            読み込むチェックポイントのパス。
        sampling_weight_scheme : str, optional, default="none"
            ホライズンごとの損失重み付け方式。['random', 'const', 'none', 'fix'] など。
        optimizer_config : dict[str, Any] | None, optional, default=None
            オプティマイザの設定 ("class_name" と "init_args" を持つ辞書)。
        lr_scheduler_config : dict[str, Any] | None, optional, default=None
            学習率スケジューラの設定 ("class_name" と "init_args" を持つ辞書)。
        **kwargs : Any
            その他の引数 (未使用)。
        """
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt
        self.train_pred_len_list = train_pred_len_list
        self.forecaster = forecaster
        self.optimizer_config = optimizer_config
        self.scheduler_config = lr_scheduler_config

        if self.optimizer_config is not None:
            print("optimizer config: ", self.optimizer_config)

        if self.scheduler_config is not None:
            print("lr_scheduler config: ", self.scheduler_config)

        self.scaler = scaler
        self.evaluator = Evaluator(quantiles_num=quantiles_num)

        # init the parapemetr for sampling
        self.sampling_weight_scheme = sampling_weight_scheme
        print(f"sampling_weight_scheme: {sampling_weight_scheme}")
        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        scaler: Scaler | None = None,
        learning_rate: float | None = None,
        no_training: bool = False,
        **kwargs: Any,
    ) -> "ProbTSForecastModule":
        """
        チェックポイントからモデルを読み込む。

        Parameters:
        ----------
        checkpoint_path : str
            チェックポイントファイルのパス。
        scaler : Scaler | None, optional, default=None
            モデルに設定するスケーラー。
        learning_rate : float | None, optional, default=None
            上書きする学習率。
        no_training : bool, optional, default=False
            学習を行わない (事前学習済みモデルとして用いる) かどうか。
        **kwargs : Any
            load_checkpoint に渡すその他の引数。

        Returns:
        ----------
        ProbTSForecastModule
            チェックポイントから復元したモデル。
        """
        model = load_checkpoint(
            self,
            checkpoint_path,
            scaler=scaler,
            learning_rate=learning_rate,
            no_training=no_training,
            **kwargs,
        )
        return model

    def training_forward(self, batch_data: ProbTSBatchData) -> torch.Tensor:
        """
        入力を正規化して損失を計算する。
        損失がホライズン次元を持つ場合は sampling_weight_scheme に従って重み付けする。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。

        Returns:
        ----------
        torch.Tensor
            学習損失。
        """
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = self.scaler.transform(
            batch_data.future_target_cdf
        )
        loss = self.forecaster.loss(batch_data)

        if len(loss.shape) > 1:
            loss_weights = get_weights(self.sampling_weight_scheme, loss.shape[1])
            loss = (
                loss_weights.detach().to(loss.device).unsqueeze(0).unsqueeze(-1) * loss
            ).sum(dim=1)
            loss = loss.mean()

        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        1 バッチ分の学習ステップを実行し、学習損失をログに記録する。

        Parameters:
        ----------
        batch : dict[str, torch.Tensor]
            データローダから得られたバッチ (フィールド名 -> テンソル)。
        batch_idx : int
            バッチのインデックス。

        Returns:
        ----------
        torch.Tensor
            学習損失。
        """
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(
        self,
        batch: dict[str, torch.Tensor],
        stage: str = "",
        dataloader_idx: int | None = None,
    ) -> dict[str, float]:
        """
        1 バッチ分の予測を行い、非正規化・正規化の両スケールで評価指標を計算・蓄積する。

        Parameters:
        ----------
        batch : dict[str, torch.Tensor]
            データローダから得られたバッチ (フィールド名 -> テンソル)。
        stage : str, optional, default=""
            評価ステージ名 ('val' または 'test')。指標名の接頭辞に使われる。
        dataloader_idx : int | None, optional, default=None
            複数データローダ使用時のデータローダのインデックス。

        Returns:
        ----------
        dict[str, float]
            ホライズンに対する評価指標 (非正規化スケール)。
        """
        batch_data = ProbTSBatchData(batch, self.device)
        pred_len = batch_data.future_target_cdf.shape[1]
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]

        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        self.batch_size.append(orin_past_data.shape[0])

        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)[
            :, :, :pred_len
        ]

        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
        )
        self.metrics_dict = update_metrics(
            metrics, stage, target_dict=self.metrics_dict
        )

        # Calculate norm metrics
        norm_metrics = self.evaluator(
            norm_future_data,
            forecasts,
            past_data=norm_past_data,
            freq=self.forecaster.freq,
        )
        self.metrics_dict = update_metrics(
            norm_metrics, stage, "norm", target_dict=self.metrics_dict
        )

        l = orin_future_data.shape[1]

        if stage != "test" and self.sampling_weight_scheme not in ["fix", "none"]:
            loss_weights = get_weights("random", l)
        else:
            loss_weights = None

        hor_metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
            loss_weights=loss_weights,
        )

        if stage == "test":
            hor_str = get_hor_str(self.forecaster.prediction_length, dataloader_idx)
            if hor_str not in self.hor_metrics:
                self.hor_metrics[hor_str] = {}
            self.hor_metrics[hor_str] = update_metrics(
                hor_metrics, stage, target_dict=self.hor_metrics[hor_str]
            )

        return hor_metrics

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> dict[str, float]:
        """
        1 バッチ分の検証ステップを実行する。

        Parameters:
        ----------
        batch : dict[str, torch.Tensor]
            データローダから得られたバッチ (フィールド名 -> テンソル)。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int | None, optional, default=None
            複数データローダ使用時のデータローダのインデックス。

        Returns:
        ----------
        dict[str, float]
            評価指標。
        """
        metrics = self.evaluate(batch, stage="val", dataloader_idx=dataloader_idx)
        return metrics

    def on_validation_epoch_start(self) -> None:
        """
        検証エポック開始時に、評価指標とバッチサイズの蓄積用変数を初期化する。
        """
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.batch_size = []

    def on_validation_epoch_end(self) -> None:
        """
        検証エポック終了時に、バッチサイズで加重平均した評価指標をログに記録する。
        """
        avg_metrics = calculate_weighted_average(self.metrics_dict, self.batch_size)
        self.log_dict(avg_metrics, prog_bar=True)

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> dict[str, float]:
        """
        1 バッチ分のテストステップを実行する。

        Parameters:
        ----------
        batch : dict[str, torch.Tensor]
            データローダから得られたバッチ (フィールド名 -> テンソル)。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int | None, optional, default=None
            複数データローダ使用時のデータローダのインデックス。

        Returns:
        ----------
        dict[str, float]
            評価指標。
        """
        metrics = self.evaluate(batch, stage="test", dataloader_idx=dataloader_idx)
        return metrics

    def on_test_epoch_start(self) -> None:
        """
        テストエポック開始時に、評価指標とバッチサイズの蓄積用変数を初期化する。
        """
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.avg_metrics = {}
        self.avg_hor_metrics = {}
        self.batch_size = []

    def on_test_epoch_end(self) -> None:
        """
        テストエポック終了時に、バッチサイズで加重平均した評価指標
        (ホライズンごとを含む) を計算し、単一ホライズンの場合はログに記録する。
        """
        if len(self.hor_metrics) > 0:
            for hor_str, metric in self.hor_metrics.items():
                self.avg_hor_metrics[hor_str] = calculate_weighted_average(
                    metric, batch_size=self.batch_size
                )
                self.avg_metrics.update(
                    calculate_weighted_average(
                        metric, batch_size=self.batch_size, hor=hor_str + "_"
                    )
                )
        else:
            self.avg_metrics = calculate_weighted_average(
                self.metrics_dict, self.batch_size
            )

        if (
            isinstance(self.forecaster.prediction_length, int)
            or len(self.forecaster.prediction_length) < 2
        ):
            self.log_dict(self.avg_metrics, logger=True)

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        1 バッチ分の予測を行う。

        Parameters:
        ----------
        batch : dict[str, torch.Tensor]
            データローダから得られたバッチ (フィールド名 -> テンソル)。
        batch_idx : int
            バッチのインデックス。

        Returns:
        ----------
        torch.Tensor
            予測サンプル。形状 [B, num_samples, L, K]。
        """
        batch_data = ProbTSBatchData(batch, self.device)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        return forecasts

    def configure_optimizers(self) -> optim.Optimizer | dict[str, Any]:
        """
        オプティマイザ (および設定されていれば学習率スケジューラ) を構成する。

        Returns:
        ----------
        optim.Optimizer | dict[str, Any]
            スケジューラ未設定時はオプティマイザ、
            設定時は "optimizer" と "lr_scheduler" を持つ辞書。
        """
        if self.optimizer_config is None:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = init_class_helper(self.optimizer_config["class_name"])
            params = self.optimizer_config["init_args"]
            optimizer = optimizer(self.parameters(), **params)

        if self.scheduler_config is not None:
            scheduler = init_class_helper(self.scheduler_config["class_name"])
            params = self.scheduler_config["init_args"]
            scheduler = scheduler(optimizer=optimizer, **params)

            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return optimizer
