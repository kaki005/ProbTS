from typing import Any, NoReturn

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from probts.data.data_manager import DataManager
from probts.data.data_wrapper import ProbTSBatchData
from probts.data.datasets.single_horizon_datasets import TransformedIterableDataset


class EmptyDataset(Dataset):
    """
    要素を持たない空のデータセット。検証セットが存在しない場合のダミーとして使用する。
    """

    def __len__(self) -> int:
        """
        データセットの要素数を返す。

        Returns:
        ----------
        int
            常に 0。
        """
        return 0

    def __getitem__(self, idx: int) -> NoReturn:
        """
        要素を取得する。空のデータセットのため常に IndexError を送出する。

        Parameters:
        ----------
        idx : int
            取得する要素のインデックス。

        Raises:
        ----------
        IndexError
            データセットが空であるため常に送出される。
        """
        raise IndexError("This dataset is empty.")


class ProbTSDataModule(pl.LightningDataModule):
    r"""
    確率的時系列データセット向けの DataModule。

    Attributes:
    ----------
    data_manager : DataManager
        データセットを管理する DataManager。
    batch_size : int
        学習時のバッチサイズ。
    test_batch_size : int
        検証・テスト・推論時のバッチサイズ。
    num_workers : int
        単一ホライズン学習時の DataLoader のワーカー数。
    dataset_train : TransformedIterableDataset
        学習用データセット。
    dataset_val : TransformedIterableDataset | dict[str, TransformedIterableDataset] | None
        検証用データセット (マルチホライズン時は予測長をキーとする辞書、検証セットが無い場合は None)。
    dataset_test : TransformedIterableDataset | dict[str, TransformedIterableDataset]
        テスト用データセット (マルチホライズン時は予測長をキーとする辞書)。
    """

    data_manager: DataManager
    batch_size: int
    test_batch_size: int
    num_workers: int
    dataset_train: TransformedIterableDataset
    dataset_val: (
        TransformedIterableDataset | dict[str, TransformedIterableDataset] | None
    )
    dataset_test: TransformedIterableDataset | dict[str, TransformedIterableDataset]

    def __init__(
        self,
        data_manager: DataManager,
        batch_size: int = 64,
        test_batch_size: int = 8,
        num_workers: int = 8,
    ) -> None:
        """
        ProbTSDataModule を初期化する。

        Parameters:
        ----------
        data_manager : DataManager
            データセットを管理する DataManager。
        batch_size : int, optional, default=64
            学習時のバッチサイズ。
        test_batch_size : int, optional, default=8
            検証・テスト・推論時のバッチサイズ。
        num_workers : int, optional, default=8
            単一ホライズン学習時の DataLoader のワーカー数。
        """
        super().__init__()
        self.data_manager = data_manager
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

        self.dataset_train = self.data_manager.train_iter_dataset
        self.dataset_val = self.data_manager.val_iter_dataset
        self.dataset_test = self.data_manager.test_iter_dataset

    def train_dataloader(self) -> DataLoader:
        """
        学習用 DataLoader を生成する。

        マルチホライズン時は可変長ホライズンをパディングする collate_fn を用いる。

        Returns:
        ----------
        DataLoader
            学習用 DataLoader。
        """
        if self.data_manager.multi_hor:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                collate_fn=self.train_collate_fn,
            )
        else:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

    def val_dataloader(self) -> DataLoader | CombinedLoader:
        """
        検証用 DataLoader を生成する。

        検証セットが無い場合は空の DataLoader を、マルチホライズン時は
        ホライズンごとの DataLoader を結合した CombinedLoader を返す。

        Returns:
        ----------
        DataLoader | CombinedLoader
            検証用 DataLoader。
        """
        # if no validation set available
        if self.dataset_val is None:
            return DataLoader(EmptyDataset(), batch_size=1)

        if self.data_manager.multi_hor:
            val_dataloader = self.combine_dataloader(self.dataset_val)
        else:
            val_dataloader = DataLoader(
                self.dataset_val, batch_size=self.test_batch_size, num_workers=1
            )
        return val_dataloader

    def test_dataloader(self) -> DataLoader | CombinedLoader:
        """
        テスト用 DataLoader を生成する。

        マルチホライズン時はホライズンごとの DataLoader を結合した CombinedLoader を返す。

        Returns:
        ----------
        DataLoader | CombinedLoader
            テスト用 DataLoader。
        """
        if self.data_manager.multi_hor:
            return self.combine_dataloader(self.dataset_test)
        else:
            return DataLoader(
                self.dataset_test, batch_size=self.test_batch_size, num_workers=1
            )

    def predict_dataloader(self) -> DataLoader:
        """
        推論用 DataLoader を生成する (テスト用データセットを使用)。

        Returns:
        ----------
        DataLoader
            推論用 DataLoader。
        """
        return DataLoader(
            self.dataset_test, batch_size=self.test_batch_size, num_workers=0
        )

    def combine_dataloader(
        self, dataset_dict: dict[str, TransformedIterableDataset]
    ) -> CombinedLoader:
        """
        ホライズンごとのデータセットから DataLoader を生成し、逐次実行の CombinedLoader に結合する。

        Parameters:
        ----------
        dataset_dict : dict[str, TransformedIterableDataset]
            予測長 (文字列) をキーとするデータセットの辞書。

        Returns:
        ----------
        CombinedLoader
            mode="sequential" で結合した CombinedLoader。
        """
        dataloader_dict = {}
        for hor in dataset_dict:
            dataloader_dict[hor] = DataLoader(
                dataset_dict[hor],
                batch_size=self.test_batch_size,
                num_workers=0,
                persistent_workers=False,
            )

        combined_loader = CombinedLoader(dataloader_dict, mode="sequential")
        return combined_loader

    def train_collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        可変ホライズン学習のため、バッチ内の各サンプルのホライズンをパディングして結合する。
        バッチ内でサンプルごとに異なる過去ウィンドウ長を持つことができる。

        過去区間は末尾寄せ、未来区間は先頭寄せでゼロパディングされる。

        Parameters:
        ----------
        batch : list[dict[str, Any]]
            サンプル (フィールド名 -> 配列) のリスト。

        Returns:
        ----------
        dict[str, Any]
            パディング済みテンソルに加え、context_length / prediction_length
            (サンプルごとの長さのリスト)、max_context_length / max_prediction_length
            を含むバッチ辞書。
        """

        past_len_list = [len(x["past_target_cdf"]) for x in batch]
        future_len_list = [len(x["future_target_cdf"]) for x in batch]

        max_past_length = max(past_len_list)
        max_future_length = max(future_len_list)
        B = len(batch)
        batch_dict = {}
        batch_dict["context_length"] = []
        batch_dict["prediction_length"] = []
        batch_dict["target_dimension_indicator"] = []

        for idx in range(len(batch)):
            local_past_len = len(batch[idx]["past_target_cdf"])
            local_future_len = len(batch[idx]["future_target_cdf"])

            for input in ProbTSBatchData.input_names_:
                K = batch[0][input].shape[-1]
                if input in [
                    "past_target_cdf",
                    "past_observed_values",
                    "past_time_feat",
                    "past_is_pad",
                ]:
                    if input not in batch_dict and input in [
                        "past_target_cdf",
                        "past_observed_values",
                        "past_time_feat",
                    ]:
                        batch_dict[input] = torch.zeros([B, max_past_length, K])
                    if input not in batch_dict and input in ["past_is_pad"]:
                        batch_dict[input] = torch.zeros([B, max_past_length])

                    batch_dict[input][idx][-local_past_len:] = torch.tensor(
                        batch[idx][input]
                    )[:local_past_len]

                elif input in [
                    "future_target_cdf",
                    "future_observed_values",
                    "future_time_feat",
                ]:
                    if input not in batch_dict:
                        batch_dict[input] = torch.zeros([B, max_future_length, K])
                    batch_dict[input][idx][:local_future_len] = torch.tensor(
                        batch[idx][input]
                    )[:local_future_len]

            batch_dict["target_dimension_indicator"].append(
                batch[idx]["target_dimension_indicator"]
            )
            batch_dict["context_length"].append(local_past_len)
            batch_dict["prediction_length"].append(local_future_len)

        batch_dict["target_dimension_indicator"] = torch.tensor(
            batch_dict["target_dimension_indicator"]
        )

        batch_dict["max_context_length"] = max_past_length
        batch_dict["max_prediction_length"] = max_future_length
        return batch_dict
