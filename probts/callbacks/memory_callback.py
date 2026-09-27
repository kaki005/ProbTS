import gc
import threading
from typing import Any

import lightning.pytorch as pl
import psutil
import torch
from lightning.pytorch.callbacks.callback import Callback


def byte2gb(x: int | float) -> float:
    """
    バイト数を GB (2**30 バイト単位) に変換する。

    Parameters:
    ----------
    x : int | float
        変換対象のバイト数。

    Returns:
    ----------
    float
        GB 単位に変換した値。
    """
    return float(x / 2**30)


class MemoryTrace:
    """
    生成時点から __exit__ 呼び出しまでの GPU / CPU メモリ使用量を計測するクラス。

    Attributes:
    ----------
    begin : float
        計測開始時点の CUDA 割り当て済みメモリ量 (GB)。
    process : psutil.Process
        現在のプロセスを表す psutil.Process。
    cpu_begin : float
        計測開始時点の CPU 常駐メモリ量 (GB)。
    peak_monitoring : bool
        CPU ピークメモリの監視スレッドを継続するかどうか。
    cpu_peak : int
        監視スレッドで観測された CPU 常駐メモリ量のピーク (バイト)。
    end : float
        計測終了時点の CUDA 割り当て済みメモリ量 (GB)。
    peak : float
        計測期間中の CUDA 割り当て済みメモリ量のピーク (GB)。
    peak_active_gb : float
        計測期間中の CUDA アクティブメモリ量のピーク (GB)。
    cuda_malloc_retires : int
        cudaMalloc のリトライ回数。
    m_cuda_ooms : int
        CUDA の OOM 発生回数。
    used : float
        計測期間中に増加した CUDA メモリ量 (end - begin を再度 GB 変換した値)。
    peaked : float
        開始時点からのピーク増加量 (peak - begin を再度 GB 変換した値)。
    max_reserved : float
        CUDA キャッシュアロケータが予約したメモリ量のピーク (GB)。
    cpu_end : int
        計測終了時点の CPU 常駐メモリ量 (バイト)。
    cpu_used : float
        計測期間中に増加した CPU メモリ量 (cpu_end - cpu_begin を GB 変換した値)。
    cpu_peaked : float
        開始時点からの CPU ピーク増加量 (cpu_peak - cpu_begin を GB 変換した値)。
    """

    begin: float
    process: psutil.Process
    cpu_begin: float
    peak_monitoring: bool
    cpu_peak: int
    end: float
    peak: float
    peak_active_gb: float
    cuda_malloc_retires: int
    m_cuda_ooms: int
    used: float
    peaked: float
    max_reserved: float
    cpu_end: int
    cpu_used: float
    cpu_peaked: float

    def __init__(self) -> None:
        """
        MemoryTrace を初期化し、メモリ計測を開始する。

        GC と CUDA キャッシュの解放、ピーク値のリセットを行った後、
        CPU ピークメモリを監視するデーモンスレッドを起動する。
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def cpu_mem_used(self) -> int:
        """
        現在のプロセスの常駐セットサイズ (RSS) メモリ量を取得する。

        Returns:
        ----------
        int
            現在のプロセスの RSS (バイト)。
        """
        return self.process.memory_info().rss

    def peak_monitor_func(self) -> None:
        """
        peak_monitoring が False になるまで CPU メモリ使用量を監視し、ピーク値を cpu_peak に記録する。
        """
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc: Any) -> None:
        """
        メモリ計測を終了し、GPU / CPU メモリ使用量の統計を属性に設定する。

        Parameters:
        ----------
        *exc : Any
            コンテキストマネージャプロトコルで渡される例外情報 (未使用)。
        """
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = byte2gb(torch.cuda.memory_allocated())
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)


class MemoryCallback(Callback):
    """
    学習・検証・テストの各エポックにおけるメモリ使用量を追跡するコールバック。

    Attributes:
    ----------
    memory_summary : dict[str, dict[str, float]]
        フェーズ ("train", "val", "test") ごとのメモリ使用量サマリ。
    train_memtrace : MemoryTrace
        学習エポック中のメモリ計測 (CUDA 利用時のみ設定)。
    val_memtrace : MemoryTrace
        検証エポック中のメモリ計測 (CUDA 利用時のみ設定)。
    test_memtrace : MemoryTrace
        テストエポック中のメモリ計測 (CUDA 利用時のみ設定)。
    """

    memory_summary: dict[str, dict[str, float]]
    train_memtrace: MemoryTrace
    val_memtrace: MemoryTrace
    test_memtrace: MemoryTrace

    def __init__(self) -> None:
        """
        MemoryCallback を初期化し、空のメモリ使用量サマリを用意する。
        """
        self.memory_summary = {"train": {}, "val": {}, "test": {}}

    def update_memory_summary(self, key: str, memtrace: MemoryTrace) -> None:
        """
        指定フェーズのメモリ使用量サマリを、既存値と新たな計測値の最大値で更新する。

        Parameters:
        ----------
        key : str
            更新対象のフェーズ ("train", "val", "test")。
        memtrace : MemoryTrace
            計測を終了した MemoryTrace。
        """
        self.memory_summary[key] = {
            "mem_peak": max(memtrace.peak, self.memory_summary[key].get("mem_peak", 0)),
            "max_reserved": max(
                memtrace.max_reserved, self.memory_summary[key].get("max_reserved", 0)
            ),
            "peak_active_gb": max(
                memtrace.peak_active_gb,
                self.memory_summary[key].get("peak_active_gb", 0),
            ),
            "cuda_malloc_retires": max(
                memtrace.cuda_malloc_retires,
                self.memory_summary[key].get("cuda_malloc_retires", 0),
            ),
            "cpu_total_peaked": max(
                memtrace.cpu_peaked + memtrace.cpu_begin,
                self.memory_summary[key].get("cpu_total_peaked", 0),
            ),
        }

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        学習エポック開始時に呼ばれ、メモリ計測を開始する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            学習対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.train_memtrace = MemoryTrace()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        学習エポック終了時に呼ばれ、メモリ計測を終了してサマリを更新する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            学習対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.train_memtrace.__exit__()
            self.update_memory_summary("train", self.train_memtrace)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        検証エポック開始時に呼ばれ、メモリ計測を開始する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            検証対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.val_memtrace = MemoryTrace()

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        検証エポック終了時に呼ばれ、メモリ計測を終了してサマリを更新する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            検証対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.val_memtrace.__exit__()
            self.update_memory_summary("val", self.val_memtrace)

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        テストエポック開始時に呼ばれ、メモリ計測を開始する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            テスト対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.test_memtrace = MemoryTrace()

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        テストエポック終了時に呼ばれ、メモリ計測を終了してサマリを更新する。

        Parameters:
        ----------
        trainer : pl.Trainer
            Lightning の Trainer。
        pl_module : pl.LightningModule
            テスト対象の LightningModule。
        """
        if torch.cuda.is_available():
            self.test_memtrace.__exit__()
            self.update_memory_summary("test", self.test_memtrace)
