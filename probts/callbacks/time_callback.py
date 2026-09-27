import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class TimeCallback(Callback):
    """
    学習・検証・テストの各バッチの計算時間を追跡するコールバック。

    Attributes:
    ----------
    time_summary : dict[str, list[float]]
        フェーズごとのバッチ処理時間 (秒) のリスト。
        キーは "train_batch_time", "val_batch_time", "test_batch_time"。
    train_start_time : float
        直近の学習バッチ開始時刻 (time.time() の値)。
    val_start_time : float
        直近の検証バッチ開始時刻 (time.time() の値)。
    test_start_time : float
        直近のテストバッチ開始時刻 (time.time() の値)。
    """

    time_summary: dict[str, list[float]]
    train_start_time: float
    val_start_time: float
    test_start_time: float

    def __init__(self) -> None:
        """
        TimeCallback を初期化し、空の計算時間サマリを用意する。
        """
        self.time_summary = {
            "train_batch_time": [],
            "val_batch_time": [],
            "test_batch_time": [],
        }

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        学習バッチ開始時に呼ばれ、開始時刻を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            学習対象の LightningModule。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        """
        self.train_start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        学習バッチ終了時に呼ばれ、バッチの処理時間を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            学習対象の LightningModule。
        outputs : STEP_OUTPUT
            training_step の出力。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        """
        self.time_summary["train_batch_time"].append(
            time.time() - self.train_start_time
        )

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        検証バッチ開始時に呼ばれ、開始時刻を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            検証対象の LightningModule。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int, optional, default=0
            データローダのインデックス。
        """
        self.val_start_time = time.time()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        検証バッチ終了時に呼ばれ、バッチの処理時間を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            検証対象の LightningModule。
        outputs : STEP_OUTPUT
            validation_step の出力。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int, optional, default=0
            データローダのインデックス。
        """
        self.time_summary["val_batch_time"].append(time.time() - self.val_start_time)

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        テストバッチ開始時に呼ばれ、開始時刻を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            テスト対象の LightningModule。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int, optional, default=0
            データローダのインデックス。
        """
        self.test_start_time = time.time()

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        テストバッチ終了時に呼ばれ、バッチの処理時間を記録する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            テスト対象の LightningModule。
        outputs : STEP_OUTPUT
            test_step の出力。
        batch : Any
            現在のバッチ。
        batch_idx : int
            バッチのインデックス。
        dataloader_idx : int, optional, default=0
            データローダのインデックス。
        """
        self.time_summary["test_batch_time"].append(time.time() - self.test_start_time)
