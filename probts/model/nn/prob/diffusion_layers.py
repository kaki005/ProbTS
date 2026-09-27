# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math

import torch
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torch import nn


def get_torch_trans(
    heads: int = 8, layers: int = 1, channels: int = 64, linear: bool = False
) -> LinearAttentionTransformer | nn.TransformerEncoder:
    """
    Transformer エンコーダ (通常版または線形注意版) を生成する。

    Parameters:
    ----------
    heads : int, optional, default=8
        アテンションヘッド数。
    layers : int, optional, default=1
        エンコーダ層の数。
    channels : int, optional, default=64
        モデル次元 (d_model)。
    linear : bool, optional, default=False
        True の場合は LinearAttentionTransformer を使用する。

    Returns:
    ----------
    LinearAttentionTransformer | nn.TransformerEncoder
        生成された Transformer エンコーダ。
    """
    if linear:
        encoder_layer = LinearAttentionTransformer(
            dim=channels,
            heads=heads,
            depth=layers,
            max_seq_len=4096,
            n_local_attn_heads=0,
        )
        return encoder_layer
    else:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    """
    Kaiming 正規分布で重みを初期化した Conv1d 層を生成する。

    Parameters:
    ----------
    in_channels : int
        入力チャネル数。
    out_channels : int
        出力チャネル数。
    kernel_size : int
        カーネルサイズ。

    Returns:
    ----------
    nn.Conv1d
        初期化済みの Conv1d 層。
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """
    拡散ステップを正弦波埋め込みと MLP によってベクトルに埋め込む層。

    Attributes:
    ----------
    embedding : torch.Tensor
        事前計算した正弦波埋め込みテーブル。shape: (max_steps, 2 * dim)。register_buffer (persistent=False) で登録。
    projection1 : nn.Linear
        埋め込みを proj_dim に射影する線形層。
    projection2 : nn.Linear
        proj_dim から proj_dim への線形層。
    """

    embedding: torch.Tensor
    projection1: nn.Linear
    projection2: nn.Linear

    def __init__(
        self, dim: int = 128, proj_dim: int | None = None, max_steps: int = 500
    ) -> None:
        """
        DiffusionEmbedding を初期化する。

        Parameters:
        ----------
        dim : int, optional, default=128
            正弦波埋め込みの周波数の数 (埋め込みテーブルの次元は 2 * dim)。
        proj_dim : int | None, optional, default=None
            出力埋め込みの次元数。None の場合は dim を使用する。
        max_steps : int, optional, default=500
            拡散ステップの最大数。
        """
        super().__init__()
        if proj_dim is None:
            proj_dim = dim
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        拡散ステップを埋め込みベクトルに変換する。

        Parameters:
        ----------
        diffusion_step : torch.Tensor
            拡散ステップのインデックス (long 型)。shape: (B,)。

        Returns:
        ----------
        torch.Tensor
            拡散ステップ埋め込み。shape: (B, proj_dim)。
        """
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim: int, max_steps: int) -> torch.Tensor:
        """
        正弦波埋め込みテーブルを構築する。

        Parameters:
        ----------
        dim : int
            周波数の数。
        max_steps : int
            拡散ステップの最大数。

        Returns:
        ----------
        torch.Tensor
            sin と cos を連結した埋め込みテーブル。shape: (max_steps, 2 * dim)。
        """
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class diff_CSDI(nn.Module):
    """
    CSDI のノイズ予測ネットワーク (時間方向・特徴方向の Transformer を持つ残差ブロックの積み重ね)。

    Attributes:
    ----------
    channels : int
        内部表現のチャネル数。
    diffusion_embedding : DiffusionEmbedding
        拡散ステップ埋め込み層。
    input_projection : nn.Conv1d
        入力を channels 次元に射影する 1x1 畳み込み。
    output_projection1 : nn.Conv1d
        出力側の 1x1 畳み込み (channels -> channels)。
    output_projection2 : nn.Conv1d
        出力側の 1x1 畳み込み (channels -> 1)。重みはゼロ初期化。
    residual_layers : nn.ModuleList
        ResidualBlock のリスト。
    """

    channels: int
    diffusion_embedding: DiffusionEmbedding
    input_projection: nn.Conv1d
    output_projection1: nn.Conv1d
    output_projection2: nn.Conv1d
    residual_layers: nn.ModuleList

    def __init__(
        self,
        channels: int,
        diffusion_embedding_dim: int,
        side_dim: int,
        num_steps: int,
        nheads: int,
        n_layers: int,
        inputdim: int = 2,
        linear: bool = False,
    ) -> None:
        """
        diff_CSDI を初期化する。

        Parameters:
        ----------
        channels : int
            内部表現のチャネル数。
        diffusion_embedding_dim : int
            拡散ステップ埋め込みの次元数。
        side_dim : int
            サイド情報 (条件情報) のチャネル数。
        num_steps : int
            拡散ステップ数。
        nheads : int
            Transformer のアテンションヘッド数。
        n_layers : int
            残差ブロックの数。
        inputdim : int, optional, default=2
            入力チャネル数。
        linear : bool, optional, default=False
            True の場合は線形注意 Transformer を使用する。
        """
        super().__init__()
        self.channels = channels

        self.diffusion_embedding = DiffusionEmbedding(
            dim=diffusion_embedding_dim, max_steps=num_steps
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    nheads=nheads,
                    linear=linear,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_step: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズを予測する。

        Parameters:
        ----------
        x : torch.Tensor
            ノイズ付き入力。shape: (B, inputdim, K, L)。
        cond_info : torch.Tensor
            サイド情報 (条件情報)。shape: (B, side_dim, K, L)。
        diffusion_step : torch.Tensor
            拡散ステップのインデックス。shape: (B,) または (1,)。

        Returns:
        ----------
        torch.Tensor
            予測されたノイズ。shape: (B, K, L)。
        """
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)

        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    """
    CSDI の残差ブロック (時間方向・特徴方向の Transformer とゲート付き活性化を持つ)。

    Attributes:
    ----------
    side_dim : int
        サイド情報 (条件情報) のチャネル数。
    diffusion_projection : nn.Linear
        拡散ステップ埋め込みを channels 次元に射影する線形層。
    cond_projection : nn.Conv1d
        サイド情報を 2 * channels 次元に射影する 1x1 畳み込み。
    mid_projection : nn.Conv1d
        中間表現を 2 * channels 次元に射影する 1x1 畳み込み。
    output_projection : nn.Conv1d
        出力を residual と skip (各 channels 次元) に射影する 1x1 畳み込み。
    time_layer : LinearAttentionTransformer | nn.TransformerEncoder
        時間方向の Transformer。
    feature_layer : LinearAttentionTransformer | nn.TransformerEncoder
        特徴方向の Transformer。
    """

    side_dim: int
    diffusion_projection: nn.Linear
    cond_projection: nn.Conv1d
    mid_projection: nn.Conv1d
    output_projection: nn.Conv1d
    time_layer: LinearAttentionTransformer | nn.TransformerEncoder
    feature_layer: LinearAttentionTransformer | nn.TransformerEncoder

    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        linear: bool = False,
    ) -> None:
        """
        ResidualBlock を初期化する。

        Parameters:
        ----------
        side_dim : int
            サイド情報 (条件情報) のチャネル数。
        channels : int
            内部表現のチャネル数。
        diffusion_embedding_dim : int
            拡散ステップ埋め込みの次元数。
        nheads : int
            Transformer のアテンションヘッド数。
        linear : bool, optional, default=False
            True の場合は線形注意 Transformer を使用する。
        """
        super().__init__()
        self.side_dim = side_dim
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(
            heads=nheads, layers=1, channels=channels, linear=linear
        )
        self.feature_layer = get_torch_trans(
            heads=nheads, layers=1, channels=channels, linear=linear
        )

    def forward_time(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        """
        時間方向 (L 軸) に Transformer を適用する。

        Parameters:
        ----------
        y : torch.Tensor
            入力。shape: (B, channel, K * L)。
        base_shape : torch.Size
            元の形状 (B, channel, K, L)。

        Returns:
        ----------
        torch.Tensor
            出力。shape: (B, channel, K * L)。L == 1 の場合は入力をそのまま返す。
        """
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        """
        特徴方向 (K 軸) に Transformer を適用する。

        Parameters:
        ----------
        y : torch.Tensor
            入力。shape: (B, channel, K * L)。
        base_shape : torch.Size
            元の形状 (B, channel, K, L)。

        Returns:
        ----------
        torch.Tensor
            出力。shape: (B, channel, K * L)。K == 1 の場合は入力をそのまま返す。
        """
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(
        self, x: torch.Tensor, cond_info: torch.Tensor, diffusion_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        残差ブロックを適用する。

        Parameters:
        ----------
        x : torch.Tensor
            入力。shape: (B, channel, K, L)。
        cond_info : torch.Tensor
            サイド情報 (条件情報)。shape: (B, side_dim, K, L)。
        diffusion_emb : torch.Tensor
            拡散ステップ埋め込み。shape: (B, diffusion_embedding_dim)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            残差接続後の出力 (shape: (B, channel, K, L)) とスキップ接続出力 (shape: (B, channel, K, L))。
        """

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
