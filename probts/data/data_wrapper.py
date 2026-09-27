from typing import Any

import torch


class ProbTSBatchData:
    """
    モデルへの入力バッチを保持し、次元の拡張・デバイス設定・パディング処理を行うラッパークラス。

    data_dict の各キーはそのまま属性として設定される。

    Attributes:
    ----------
    input_names_ : list[str]
        モデル入力として期待されるフィールド名のリスト (クラス変数)。
    target_dimension_indicator : torch.Tensor | None
        ターゲット次元のインジケータ。
    past_time_feat : torch.Tensor | None
        過去区間の時間特徴量。
    past_target_cdf : torch.Tensor | None
        過去区間のターゲット値。
    past_observed_values : torch.Tensor | None
        過去区間の観測値インジケータ。
    past_is_pad : torch.Tensor | None
        過去区間のパディングインジケータ。
    future_time_feat : torch.Tensor | None
        未来区間の時間特徴量。
    future_target_cdf : torch.Tensor | None
        未来区間のターゲット値。
    future_observed_values : torch.Tensor | None
        未来区間の観測値インジケータ。
    context_length : int | list[int] | None
        コンテキスト長 (可変ホライズン学習時はサンプルごとのリスト)。
    prediction_length : int | list[int] | None
        予測長 (可変ホライズン学習時はサンプルごとのリスト)。
    max_context_length : int | None
        バッチ内の最大コンテキスト長。
    max_prediction_length : int | None
        バッチ内の最大予測長。
    device : torch.device | str
        テンソルを配置するデバイス。
    """

    input_names_: list[str] = [
        "target_dimension_indicator",
        "past_time_feat",
        "past_target_cdf",
        "past_observed_values",
        "past_is_pad",
        "future_time_feat",
        "future_target_cdf",
        "future_observed_values",
    ]
    target_dimension_indicator: torch.Tensor | None
    past_time_feat: torch.Tensor | None
    past_target_cdf: torch.Tensor | None
    past_observed_values: torch.Tensor | None
    past_is_pad: torch.Tensor | None
    future_time_feat: torch.Tensor | None
    future_target_cdf: torch.Tensor | None
    future_observed_values: torch.Tensor | None
    context_length: int | list[int] | None
    prediction_length: int | list[int] | None
    max_context_length: int | None
    max_prediction_length: int | None
    device: torch.device | str

    def __init__(self, data_dict: dict[str, Any], device: torch.device | str) -> None:
        """
        ProbTSBatchData を初期化する。

        data_dict の内容を属性に設定し、単変量データの次元拡張、デバイス設定、
        欠損入力の補完 (None)、パディングに基づく観測値の調整を行う。

        Parameters:
        ----------
        data_dict : dict[str, Any]
            バッチデータの辞書 (フィールド名 -> テンソル等)。
        device : torch.device | str
            テンソルを配置するデバイス。
        """
        # Initialize attributes from the provided data dictionary
        self.__dict__.update(data_dict)
        self.__dict__["context_length"] = data_dict.get("context_length", None)
        self.__dict__["prediction_length"] = data_dict.get("prediction_length", None)
        self.__dict__["max_context_length"] = data_dict.get("max_context_length", None)
        self.__dict__["max_prediction_length"] = data_dict.get(
            "max_prediction_length", None
        )

        # Expand dimensions for univariate data
        if len(self.__dict__["past_target_cdf"].shape) == 2:
            self._expand_dimensions()

        # Set tensors to the specified device
        self._set_device(device)
        # Fill missing inputs with None
        self._ensure_all_inputs_present()
        # Process padding for observed values
        self._process_padding()

    def _ensure_all_inputs_present(self) -> None:
        """
        期待されるすべての入力がデータに存在することを保証する (存在しないものは None を設定する)。
        """
        for input in self.input_names_:
            if input not in self.__dict__:
                self.__dict__[input] = None

    def _set_device(self, device: torch.device | str) -> None:
        """
        すべてのテンソルを指定デバイスへ移動し、device 属性を設定する。

        Parameters:
        ----------
        device : torch.device | str
            テンソルを配置するデバイス。
        """
        for k, v in self.__dict__.items():
            if v is not None and torch.is_tensor(v):
                v.to(device)
        self.device = device

    def _expand_dimensions(self) -> None:
        """
        必要に応じてターゲット関連テンソルの次元を拡張する (単変量データ向け)。
        """
        self.__dict__["target_dimension_indicator"] = self.__dict__[
            "target_dimension_indicator"
        ][:, :1]
        for input in [
            "past_target_cdf",
            "past_observed_values",
            "future_target_cdf",
            "future_observed_values",
        ]:
            self.__dict__[input] = self.__dict__[input].unsqueeze(-1)

    def _process_padding(self) -> None:
        """
        パディングインジケータ (past_is_pad) に基づいて観測値インジケータを調整する。
        """
        if self.__dict__["past_is_pad"] is not None:
            self.__dict__["past_observed_values"] = torch.min(
                self.__dict__["past_observed_values"],
                1 - self.__dict__["past_is_pad"].unsqueeze(-1),
            )
