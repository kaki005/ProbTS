from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from probts.data.data_utils.data_scaler import TemporalScaler
from probts.utils import weighted_average

if TYPE_CHECKING:
    from probts.data.data_wrapper import ProbTSBatchData


class Forecaster(nn.Module):
    """
    全ての予測モデルの基底クラス。入力系列・ラグ特徴量・特徴量インデックス埋め込み・
    時間特徴量の構築など、共通の入力処理を提供する。

    Attributes:
    ----------
    context_length : list[int] | int
        コンテキスト長 (またはその候補リスト)。
    prediction_length : list[int] | int
        予測長 (またはその候補リスト)。
    max_context_length : int
        最大コンテキスト長。
    max_prediction_length : int
        最大予測長。
    target_dim : int
        ターゲット変数の次元数。
    freq : str
        データの頻度 (例: 'H' は毎時, 'D' は毎日)。
    use_lags : bool
        ラグ特徴量を入力に用いるかどうか。
    use_feat_idx_emb : bool
        特徴量インデックス埋め込みを入力に用いるかどうか。
    use_time_feat : bool
        時間特徴量を入力に用いるかどうか。
    feat_idx_emb_dim : int
        特徴量インデックス埋め込みの次元数。
    time_feat_dim : int
        時間特徴量の次元数。
    autoregressive : bool
        自己回帰的に予測するかどうか。
    no_training : bool
        学習を行わない (ゼロショット / 統計モデル等) かどうか。
    use_scaling : bool
        TemporalScaler によるスケーリングを行うかどうか。
    dataset : str | None
        データセット名。
    lags_list : list[int]
        使用するラグのリスト。
    scaler : TemporalScaler | None
        スケーラー (use_scaling が True の場合のみ)。
    lags_dim : int
        ラグ特徴量の次元数 (len(lags_list) * target_dim)。
    feat_idx_emb : nn.Embedding | None
        特徴量インデックス埋め込み層 (use_feat_idx_emb が True の場合のみ)。
    input_size : int
        モデル入力の特徴量次元数。
    """

    context_length: list[int] | int
    prediction_length: list[int] | int
    max_context_length: int
    max_prediction_length: int
    target_dim: int
    freq: str
    use_lags: bool
    use_feat_idx_emb: bool
    use_time_feat: bool
    feat_idx_emb_dim: int
    time_feat_dim: int
    autoregressive: bool
    no_training: bool
    use_scaling: bool
    dataset: str | None
    lags_list: list[int]
    scaler: TemporalScaler | None
    lags_dim: int
    feat_idx_emb: nn.Embedding | None
    input_size: int

    def __init__(
        self,
        target_dim: int,
        context_length: list[int] | int,
        prediction_length: list[int] | int,
        freq: str,
        use_lags: bool = False,
        use_feat_idx_emb: bool = False,
        use_time_feat: bool = False,
        lags_list: list[int] = [],
        feat_idx_emb_dim: int = 1,
        time_feat_dim: int = 1,
        use_scaling: bool = False,
        autoregressive: bool = False,
        no_training: bool = False,
        dataset: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Forecaster を初期化する。

        Parameters:
        ----------
        target_dim : int
            ターゲット変数の次元数。
        context_length : list[int] | int
            コンテキスト長 (またはその候補リスト)。
        prediction_length : list[int] | int
            予測長 (またはその候補リスト)。
        freq : str
            データの頻度 (例: 'H' は毎時, 'D' は毎日)。
        use_lags : bool, optional, default=False
            ラグ特徴量を入力に用いるかどうか。
        use_feat_idx_emb : bool, optional, default=False
            特徴量インデックス埋め込みを入力に用いるかどうか。
        use_time_feat : bool, optional, default=False
            時間特徴量を入力に用いるかどうか。
        lags_list : list[int], optional, default=[]
            使用するラグのリスト。
        feat_idx_emb_dim : int, optional, default=1
            特徴量インデックス埋め込みの次元数。
        time_feat_dim : int, optional, default=1
            時間特徴量の次元数。
        use_scaling : bool, optional, default=False
            TemporalScaler によるスケーリングを行うかどうか。
        autoregressive : bool, optional, default=False
            自己回帰的に予測するかどうか。
        no_training : bool, optional, default=False
            学習を行わないかどうか。
        dataset : str | None, optional, default=None
            データセット名。
        **kwargs : Any
            その他の引数 (未使用)。
        """
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length

        if isinstance(self.context_length, list):
            self.max_context_length = max(self.context_length)
        else:
            self.max_context_length = self.context_length

        if isinstance(self.prediction_length, list):
            self.max_prediction_length = max(self.prediction_length)
        else:
            self.max_prediction_length = self.prediction_length

        self.target_dim = target_dim
        self.freq = freq
        self.use_lags = use_lags
        self.use_feat_idx_emb = use_feat_idx_emb
        self.use_time_feat = use_time_feat
        self.feat_idx_emb_dim = feat_idx_emb_dim
        self.time_feat_dim = time_feat_dim
        self.autoregressive = autoregressive
        self.no_training = no_training
        self.use_scaling = use_scaling
        self.dataset = dataset
        # Lag parameters
        self.lags_list = lags_list
        if self.use_scaling:
            self.scaler = TemporalScaler()
        else:
            self.scaler = None

        self.lags_dim = len(self.lags_list) * target_dim

        if use_feat_idx_emb:
            self.feat_idx_emb = nn.Embedding(
                num_embeddings=self.target_dim, embedding_dim=self.feat_idx_emb_dim
            )
        else:
            self.feat_idx_emb = None

        self.input_size = self.get_input_size()

    @property
    def name(self) -> str:
        """
        モデル名 (クラス名) を返す。

        Returns:
        ----------
        str
            クラス名。
        """
        return self.__class__.__name__

    def get_input_size(self) -> int:
        """
        設定に基づいてモデル入力の特徴量次元数を計算する。

        Returns:
        ----------
        int
            入力特徴量の次元数 (ターゲット / ラグ次元 + 特徴量インデックス埋め込み + 時間特徴量)。
        """
        input_size = self.target_dim if not self.use_lags else self.lags_dim
        if self.use_feat_idx_emb:
            input_size += self.use_feat_idx_emb * self.target_dim
        if self.use_time_feat:
            input_size += self.time_feat_dim
        return input_size

    def get_lags(
        self, sequence: torch.Tensor, lags_list: list[int], lags_length: int = 1
    ) -> torch.Tensor:
        """
        形状 (B, L, C) の系列から複数のラグを取り出し、形状 (B, L', C*N) に変換する。
        ここで L' = lags_length, N = len(lags_list) である。

        Parameters:
        ----------
        sequence : torch.Tensor
            入力系列。形状 (B, L, C)。
        lags_list : list[int]
            取り出すラグのリスト。
        lags_length : int, optional, default=1
            各ラグで取り出す長さ L'。

        Returns:
        ----------
        torch.Tensor
            ラグ特徴量を連結したテンソル。形状 (B, L', C*N)。
        """
        assert max(lags_list) + lags_length <= sequence.shape[1]

        lagged_values = []
        for lag_index in lags_list:
            begin_index = -lag_index - lags_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_value = sequence[:, begin_index:end_index, ...]
            if self.use_scaling:
                lagged_value = lagged_value / self.scaler.scale
            lagged_values.append(lagged_value)
        return torch.cat(lagged_values, dim=-1)

    def get_input_sequence(
        self,
        past_target_cdf: torch.Tensor,
        future_target_cdf: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        モードに応じて入力ターゲット系列 (またはラグ特徴量) を構築する。

        Parameters:
        ----------
        past_target_cdf : torch.Tensor
            過去区間のターゲット値。形状 [B, L, K]。
        future_target_cdf : torch.Tensor
            未来区間のターゲット値。形状 [B, L_f, K]。
        mode : str
            入力モード。['all', 'encode', 'decode'] のいずれか。

        Returns:
        ----------
        torch.Tensor
            入力系列。形状 [B, L, K] (use_lags 時は [B, L, n_lags*K])。
        """
        if mode == "all":
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            seq_length = self.max_context_length + self.max_prediction_length
        elif mode == "encode":
            sequence = past_target_cdf
            seq_length = self.max_context_length
        elif mode == "decode":
            sequence = past_target_cdf
            seq_length = 1
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

        if self.use_lags:
            input_seq = self.get_lags(sequence, self.lags_list, seq_length)
        else:
            input_seq = sequence[:, -seq_length:, ...]
            if self.use_scaling:
                input_seq = input_seq / self.scaler.scale
        return input_seq

    def get_input_feat_idx_emb(
        self, target_dimension_indicator: torch.Tensor, input_length: int
    ) -> torch.Tensor:
        """
        特徴量インデックス埋め込みを計算し、時間方向に展開する。

        Parameters:
        ----------
        target_dimension_indicator : torch.Tensor
            ターゲット次元のインジケータ。形状 [B, K]。
        input_length : int
            入力系列の長さ L。

        Returns:
        ----------
        torch.Tensor
            特徴量インデックス埋め込み。形状 [B, L, K*D]。
        """
        input_feat_idx_emb = self.feat_idx_emb(target_dimension_indicator)  # [B K D]

        input_feat_idx_emb = (
            input_feat_idx_emb.unsqueeze(1)
            .expand(-1, input_length, -1, -1)
            .reshape(-1, input_length, self.target_dim * self.feat_idx_emb_dim)
        )
        return input_feat_idx_emb  # [B L K*D]

    def get_input_time_feat(
        self,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        モードに応じて入力時間特徴量を構築する。

        Parameters:
        ----------
        past_time_feat : torch.Tensor
            過去区間の時間特徴量。形状 [B, L, Dt]。
        future_time_feat : torch.Tensor
            未来区間の時間特徴量。形状 [B, L_f, Dt]。
        mode : str
            入力モード。['all', 'encode', 'decode'] のいずれか。

        Returns:
        ----------
        torch.Tensor
            入力時間特徴量。形状 [B, L, Dt]。
        """
        if mode == "all":
            time_feat = torch.cat(
                (past_time_feat[:, -self.max_context_length :, ...], future_time_feat),
                dim=1,
            )
        elif mode == "encode":
            time_feat = past_time_feat[:, -self.max_context_length :, ...]
        elif mode == "decode":
            time_feat = future_time_feat
        return time_feat

    def get_inputs(self, batch_data: "ProbTSBatchData", mode: str) -> torch.Tensor:
        """
        入力系列・特徴量インデックス埋め込み・時間特徴量を連結してモデル入力を構築する。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。
        mode : str
            入力モード。['all', 'encode', 'decode'] のいずれか。

        Returns:
        ----------
        torch.Tensor
            float32 のモデル入力。形状 [B, L, input_size]。
        """
        inputs_list = []

        input_seq = self.get_input_sequence(
            batch_data.past_target_cdf, batch_data.future_target_cdf, mode=mode
        )
        input_length = input_seq.shape[1]  # [B L n_lags*K]
        inputs_list.append(input_seq)

        if self.use_feat_idx_emb:
            input_feat_idx_emb = self.get_input_feat_idx_emb(
                batch_data.target_dimension_indicator, input_length
            )  # [B L K*D]
            inputs_list.append(input_feat_idx_emb)

        if self.use_time_feat:
            input_time_feat = self.get_input_time_feat(
                batch_data.past_time_feat, batch_data.future_time_feat, mode=mode
            )  # [B L Dt]
            inputs_list.append(input_time_feat)
        return torch.cat(inputs_list, dim=-1).to(dtype=torch.float32)

    def get_scale(self, batch_data: "ProbTSBatchData") -> None:
        """
        過去区間のターゲット値と観測値インジケータからスケーラーを学習する。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。
        """
        self.scaler.fit(
            batch_data.past_target_cdf[:, -self.max_context_length :, ...],
            batch_data.past_observed_values[:, -self.max_context_length :, ...],
        )

    def get_weighted_loss(
        self, batch_data: "ProbTSBatchData", loss: torch.Tensor
    ) -> torch.Tensor:
        """
        未来区間の観測値インジケータで重み付けした損失の平均を計算する。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。
        loss : torch.Tensor
            要素ごとの損失。

        Returns:
        ----------
        torch.Tensor
            時間方向 (dim=1) に重み付き平均した損失。
        """
        observed_values = batch_data.future_observed_values
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        return loss

    def loss(self, batch_data: "ProbTSBatchData") -> torch.Tensor:
        """
        学習損失を計算する (サブクラスで実装する)。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。

        Returns:
        ----------
        torch.Tensor
            学習損失。
        """
        raise NotImplementedError

    def forecast(
        self,
        batch_data: "ProbTSBatchData | None" = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """
        予測を行う (サブクラスで実装する)。

        Parameters:
        ----------
        batch_data : ProbTSBatchData | None, optional, default=None
            入力バッチデータ。
        num_samples : int | None, optional, default=None
            生成する予測サンプル数。

        Returns:
        ----------
        torch.Tensor
            予測結果。形状 [B, num_samples, L, K]。
        """
        raise NotImplementedError
