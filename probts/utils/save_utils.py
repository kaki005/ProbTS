import importlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from probts.model.forecaster import Forecaster

if TYPE_CHECKING:
    from probts.data.data_utils.data_scaler import Scaler
    from probts.model.forecast_module import ProbTSForecastModule


def update_metrics(
    new_metrics: dict[str, Any],
    stage: str,
    key: str = "",
    target_dict: dict[str, list[Any]] = {},
) -> dict[str, list[Any]]:
    """
    新しく計算された指標を、接頭辞付きのキーで target_dict に追記する。

    キーは "{stage}_{metric_name}" (key 指定時は "{stage}_{key}_{metric_name}") となる。

    Parameters:
    ----------
    new_metrics : dict[str, Any]
        指標名をキー、指標値 (スカラーまたはリスト) を値とする辞書。
    stage : str
        ステージ名 (例: "val", "test")。
    key : str, optional, default=""
        追加の接頭辞 (例: "norm")。
    target_dict : dict[str, list[Any]], optional, default={}
        指標を追記する辞書 (インプレースで更新される)。

    Returns:
    ----------
    dict[str, list[Any]]
        更新後の target_dict。
    """
    prefix = stage if key == "" else f"{stage}_{key}"
    for metric_name, metric_value in new_metrics.items():
        metric_key = f"{prefix}_{metric_name}"
        if metric_key not in target_dict:
            target_dict[metric_key] = []

        if isinstance(metric_value, list):
            target_dict[metric_key] = target_dict[metric_key] + metric_value
        else:
            target_dict[metric_key].append(metric_value)

    return target_dict


def calculate_average(metrics_dict: dict[str, Any], hor: str = "") -> dict[str, Any]:
    """
    各指標の値の単純平均を計算する。

    Parameters:
    ----------
    metrics_dict : dict[str, Any]
        指標名をキー、指標値のリストを値とする辞書。
    hor : str, optional, default=""
        キーに付与するホライズン接頭辞 (空でなければ "{hor}/" が付与される)。

    Returns:
    ----------
    dict[str, Any]
        接頭辞付き指標名をキー、平均値を値とする辞書。
    """
    metrics = {}
    if hor != "":
        hor = hor + "/"

    for key, value in metrics_dict.items():
        metrics[hor + key] = np.mean(value)
    return metrics


def calculate_weighted_average(
    metrics_dict: dict[str, Any], batch_size: list[int], hor: str = ""
) -> dict[str, Any]:
    """
    各指標の値をバッチサイズで重み付けした平均を計算する。

    Parameters:
    ----------
    metrics_dict : dict[str, Any]
        指標名をキー、バッチごとの指標値のリストを値とする辞書。
    batch_size : list[int]
        各バッチのバッチサイズのリスト (重みとして使用)。
    hor : str, optional, default=""
        キーに付与する接頭辞。

    Returns:
    ----------
    dict[str, Any]
        接頭辞付き指標名をキー、重み付き平均値を値とする辞書。
    """
    metrics = {}
    for key, value in metrics_dict.items():
        metrics[hor + key] = np.sum(value * np.array(batch_size)) / np.sum(batch_size)
    return metrics


def save_point_error(
    target: np.ndarray,
    predict: np.ndarray,
    input_dict: dict[str, dict[str, list[np.ndarray]]],
    hor_str: str,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    点予測の絶対誤差・正解値・予測値をホライズンごとに input_dict へ記録する。

    Parameters:
    ----------
    target : np.ndarray
        正解値。
    predict : np.ndarray
        予測値 (target と同じ形状)。
    input_dict : dict[str, dict[str, list[np.ndarray]]]
        ホライズン文字列をキーとし、"MAE", "target", "forecast" のリストを持つ辞書 (インプレースで更新される)。
    hor_str : str
        ホライズンを表す文字列。

    Returns:
    ----------
    dict[str, dict[str, list[np.ndarray]]]
        更新後の input_dict。
    """
    if hor_str not in input_dict:
        input_dict[hor_str] = {"MAE": [], "target": [], "forecast": []}

    abs_error = np.abs(target - predict)

    input_dict[hor_str]["MAE"].append(abs_error)
    input_dict[hor_str]["target"].append(target)
    input_dict[hor_str]["forecast"].append(predict)
    return input_dict


def load_checkpoint(
    Model: "type[ProbTSForecastModule]",
    checkpoint_path: str,
    scaler: "Scaler | None" = None,
    learning_rate: float | None = None,
    no_training: bool = False,
    **kwargs: Any,
) -> "ProbTSForecastModule":
    """
    チェックポイントから forecaster とモデルを復元する。

    Parameters:
    ----------
    Model : type[ProbTSForecastModule]
        インスタンス化するモデルクラス。
    checkpoint_path : str
        チェックポイントファイルのパス。
    scaler : Scaler | None, optional, default=None
        データの正規化に用いる scaler。
    learning_rate : float | None, optional, default=None
        学習率。None の場合はチェックポイントのハイパーパラメータ (なければ 1e-3) を使用する。
    no_training : bool, optional, default=False
        学習を行わない (推論のみ) かどうか。
    **kwargs : Any
        forecaster およびモデルのコンストラクタに渡す追加の引数。

    Returns:
    ----------
    ProbTSForecastModule
        state_dict を読み込んだモデルインスタンス。
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
    # Extract the arguments for the forecaster
    forecaster_args = checkpoint["hyper_parameters"]["forecaster"]

    if isinstance(forecaster_args, Forecaster):
        forecaster = forecaster_args
    else:
        module_path, class_name = forecaster_args["class_path"].rsplit(".", 1)
        forecaster_class = getattr(importlib.import_module(module_path), class_name)

        # Add any missing required arguments
        forecaster_args = forecaster_args["init_args"]
        forecaster_args.update(kwargs)

        # Create the forecaster
        forecaster = forecaster_class(**forecaster_args)

    forecaster.no_training = no_training

    if learning_rate is None:
        learning_rate = checkpoint["hyper_parameters"].get("learning_rate", 1e-3)

    # Create the model instance
    model = Model(
        forecaster=forecaster,
        scaler=scaler,
        num_samples=checkpoint["hyper_parameters"].get("num_samples", 100),
        learning_rate=learning_rate,
        quantiles_num=checkpoint["hyper_parameters"].get("quantiles_num", 10),
        load_from_ckpt=checkpoint["hyper_parameters"].get("load_from_ckpt", None),
        **kwargs,  # Pass additional arguments here
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def get_hor_str(prediction_length: int | list[int], dataloader_idx: int | None) -> str:
    """
    予測長とデータローダーのインデックスから、ホライズンを表す文字列を取得する。

    Parameters:
    ----------
    prediction_length : int | list[int]
        予測長、または予測長のリスト。
    dataloader_idx : int | None
        データローダーのインデックス。None でなければ prediction_length[dataloader_idx] を使用する。

    Returns:
    ----------
    str
        ホライズンを表す文字列。
    """
    if dataloader_idx is not None:
        hor_str = str(prediction_length[dataloader_idx])
    elif type(prediction_length) == list:
        hor_str = str(prediction_length[0])
    else:
        hor_str = str(prediction_length)
    return hor_str


def save_exp_summary(pl_module: Any, inference: bool = False) -> None:
    """
    パラメータ数・メモリ使用量・実行時間などの実験サマリを JSON ファイルに保存する。

    Parameters:
    ----------
    pl_module : Any
        model_summary_callback, memory_callback, time_callback, trainer, model, save_dict を持つオブジェクト (通常は ProbTSCli)。
    inference : bool, optional, default=False
        推論のみの実行かどうか。True の場合 inference_summary.json、False の場合 summary.json に保存する。
    """
    exp_summary = {}

    model_summary = pl_module.model_summary_callback._summary(
        pl_module.trainer, pl_module.model
    )
    exp_summary["total_parameters"] = model_summary.total_parameters
    exp_summary["trainable_parameters"] = model_summary.trainable_parameters
    exp_summary["model_size"] = model_summary.model_size

    memory_summary = pl_module.memory_callback.memory_summary
    exp_summary["memory_summary"] = memory_summary

    time_summary = pl_module.time_callback.time_summary
    exp_summary["time_summary"] = time_summary
    for batch_key, batch_time in time_summary.items():
        if len(batch_time) > 0:
            exp_summary[f"mean_{batch_key}"] = sum(batch_time) / len(batch_time)

    exp_summary["sampling_weight_scheme"] = pl_module.model.sampling_weight_scheme

    if inference:
        summary_save_path = f"{pl_module.save_dict}/inference_summary.json"
    else:
        summary_save_path = f"{pl_module.save_dict}/summary.json"

    with open(summary_save_path, "w") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"Summary saved to {summary_save_path}")


def save_csv(
    save_dict: str, model: "ProbTSForecastModule", context_length: int
) -> None:
    """
    テスト結果 (ホライズンごとの平均指標) を CSV ファイルに保存する。

    Parameters:
    ----------
    save_dict : str
        CSV を保存するディレクトリのパス。
    model : ProbTSForecastModule
        avg_hor_metrics / avg_metrics を保持するテスト済みモデル。
    context_length : int
        コンテキスト長 (学習なしの場合にファイル名に使用)。
    """
    if len(model.avg_hor_metrics) > 0:
        horizon_list = []
        for horizon in model.avg_hor_metrics:
            horizon_dict = model.avg_hor_metrics[str(horizon)]
            horizon_dict["horizon"] = horizon
            horizon_list.append(horizon_dict)

        df = pd.DataFrame(horizon_list)

    else:
        df = pd.DataFrame([model.avg_metrics])

    if not model.forecaster.no_training:
        test_result_file = "horizons_results"
    else:
        test_result_file = f"testctx_{context_length}_horizons_results"

    df.to_csv(f"{save_dict}/{test_result_file}.csv", index="idx")
    print("horizons result saved to ", f"{save_dict}/{test_result_file}.csv")
