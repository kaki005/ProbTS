from typing import Any

import numpy as np
import torch

from .metrics import *


class Evaluator:
    """
    確率的予測の評価指標 (MSE, ND, CRPS, MASE など) を計算する評価器。

    Attributes:
    ----------
    quantiles : np.ndarray
        評価に用いる分位点の配列 (0 を除く 1/quantiles_num 刻み)。
    ignore_invalid_values : bool
        NaN/Inf などの無効値をマスクして無視するかどうか。
    smooth : bool
        平滑化を行うかどうか (現在は未使用)。
    """

    quantiles: np.ndarray
    ignore_invalid_values: bool
    smooth: bool

    def __init__(self, quantiles_num: int = 10, smooth: bool = False) -> None:
        """
        Evaluator を初期化する。

        Parameters:
        ----------
        quantiles_num : int, optional, default=10
            分位点の分割数。分位点は [1/quantiles_num, ..., (quantiles_num-1)/quantiles_num] となる。
        smooth : bool, optional, default=False
            平滑化を行うかどうか (現在は未使用)。
        """
        self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        self.ignore_invalid_values = True
        self.smooth = smooth

    def loss_name(self, q: float) -> str:
        """
        分位点損失の指標名を返す。

        Parameters:
        ----------
        q : float
            分位点。

        Returns:
        ----------
        str
            指標名 (例: "QuantileLoss[0.1]")。
        """
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q: float) -> str:
        """
        重み付き分位点損失の指標名を返す。

        Parameters:
        ----------
        q : float
            分位点。

        Returns:
        ----------
        str
            指標名 (例: "wQuantileLoss[0.1]")。
        """
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q: float) -> str:
        """
        coverage の指標名を返す。

        Parameters:
        ----------
        q : float
            分位点。

        Returns:
        ----------
        str
            指標名 (例: "Coverage[0.1]")。
        """
        return f"Coverage[{q}]"

    def get_sequence_metrics(
        self,
        targets: np.ndarray,
        forecasts: np.ndarray,
        seasonal_error: np.ndarray | None = None,
        samples_dim: int = 1,
        loss_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        単一系列に対する各種評価指標を計算する。

        Parameters:
        ----------
        targets : np.ndarray
            正解値。形状 (1, prediction_length, target_dim)。
        forecasts : np.ndarray
            サンプル予測値。形状 (1, num_samples, prediction_length, target_dim)。
        seasonal_error : np.ndarray | None, optional, default=None
            季節性誤差。None の場合 MASE は計算しない。
        samples_dim : int, optional, default=1
            forecasts におけるサンプル次元の軸。
        loss_weights : torch.Tensor | None, optional, default=None
            予測ホライズン方向の重み。形状 (prediction_length,)。None の場合 weighted_ND は ND と同じになる。

        Returns:
        ----------
        dict[str, Any]
            指標名をキー、指標値を値とする辞書。
        """
        mean_forecasts = forecasts.mean(axis=samples_dim)
        median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)
        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "abs_error": abs_error(targets, median_forecasts),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
        }

        if seasonal_error is not None:
            metrics["MASE"] = mase(targets, median_forecasts, seasonal_error)

        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]

        # calculate weighted loss
        if loss_weights is not None:
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            loss_weights = loss_weights.detach().unsqueeze(0).unsqueeze(-1).numpy()
            weighted_ND = loss_weights * nd
            metrics["weighted_ND"] = np.sum(weighted_ND)
        else:
            metrics["weighted_ND"] = metrics["ND"]

        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = np.sum(quantile_loss(targets, q_forecasts, q))
            metrics[self.weighted_loss_name(q)] = (
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            )
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)

        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantiles]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantiles]
        )

        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantiles
            ]
        )
        return metrics

    def get_metrics(
        self,
        targets: np.ndarray,
        forecasts: np.ndarray,
        seasonal_error: np.ndarray | None = None,
        samples_dim: int = 1,
        loss_weights: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        バッチ内の各系列について指標を計算し、系列間で平均した指標を返す。

        Parameters:
        ----------
        targets : np.ndarray
            正解値。形状 (batch_size, prediction_length, target_dim)。
        forecasts : np.ndarray
            サンプル予測値。形状 (batch_size, num_samples, prediction_length, target_dim)。
        seasonal_error : np.ndarray | None, optional, default=None
            系列ごとの季節性誤差。形状 (batch_size, 1, target_dim)。None の場合 MASE は計算しない。
        samples_dim : int, optional, default=1
            forecasts におけるサンプル次元の軸。
        loss_weights : torch.Tensor | None, optional, default=None
            予測ホライズン方向の重み。形状 (prediction_length,)。

        Returns:
        ----------
        dict[str, Any]
            指標名をキー、系列平均した指標値を値とする辞書。
        """
        metrics = {}
        seq_metrics = {}

        # Calculate metrics for each sequence
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(seasonal_error[i], axis=0)
                if seasonal_error is not None
                else None,
                samples_dim,
                loss_weights,
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)

        for metric_name, metric_values in seq_metrics.items():
            metrics[metric_name] = np.mean(metric_values)
        return metrics

    @property
    def selected_metrics(self) -> list[str]:
        """
        最終的に出力する指標名のリストを返す。

        Returns:
        ----------
        list[str]
            出力対象の指標名のリスト。
        """
        return ["ND", "weighted_ND", "CRPS", "NRMSE", "MSE", "MASE"]

    def __call__(
        self,
        targets: torch.Tensor | np.ndarray,
        forecasts: torch.Tensor | np.ndarray,
        past_data: torch.Tensor | np.ndarray,
        freq: str,
        loss_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        予測結果を評価し、選択された指標 (および全変数和に対する指標) を返す。

        Parameters:
        ----------
        targets : torch.Tensor | np.ndarray
            正解値。形状 (batch_size, prediction_length, target_dim)。
        forecasts : torch.Tensor | np.ndarray
            サンプル予測値。形状 (batch_size, num_samples, prediction_length, target_dim)。
        past_data : torch.Tensor | np.ndarray
            季節性誤差の計算に用いる過去データ。形状 (batch_size, history_length, target_dim)。
        freq : str
            データの頻度 (例: 'H', 'D')。
        loss_weights : torch.Tensor | None, optional, default=None
            予測ホライズン方向の重み。形状 (prediction_length,)。

        Returns:
        ----------
        dict[str, float]
            指標名をキー、指標値を値とする辞書 (変数和に対する指標は "-Sum" 接尾辞付き)。
        """

        targets = process_tensor(targets)
        forecasts = process_tensor(forecasts)
        past_data = process_tensor(past_data)

        if self.ignore_invalid_values:
            targets = np.ma.masked_invalid(targets)
            forecasts = np.ma.masked_invalid(forecasts)

        seasonal_error = calculate_seasonal_error(past_data, freq)

        metrics = self.get_metrics(
            targets,
            forecasts,
            seasonal_error=seasonal_error,
            samples_dim=1,
            loss_weights=loss_weights,
        )
        metrics_sum = self.get_metrics(
            targets.sum(axis=-1), forecasts.sum(axis=-1), samples_dim=1
        )

        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
            output_metrics[k] = metrics[k]
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = metrics_sum[k]
        return output_metrics


def process_tensor(targets: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    torch.Tensor を numpy 配列に変換する (numpy 配列はそのまま返す)。

    Parameters:
    ----------
    targets : torch.Tensor | np.ndarray
        変換対象のテンソルまたは配列。

    Returns:
    ----------
    np.ndarray
        numpy 配列。

    Raises:
    ----------
    TypeError
        targets が torch.Tensor でも numpy.ndarray でもない場合。
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    elif isinstance(targets, np.ndarray):
        pass
    else:
        raise TypeError("targets must be a torch.Tensor or a numpy.ndarray")
    return targets
