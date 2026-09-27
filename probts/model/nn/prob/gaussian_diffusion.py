# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math
from functools import partial
from inspect import isfunction
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from probts.model.nn.prob.diffusion_layers import DiffusionEmbedding


def default(val: Any, d: Any) -> Any:
    """
    val が None でなければ val を、None なら d (関数の場合はその呼び出し結果) を返す。

    Parameters:
    ----------
    val : Any
        優先して返す値。
    d : Any
        val が None の場合に用いるデフォルト値、またはデフォルト値を返す関数。

    Returns:
    ----------
    Any
        val または d から得られた値。
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(
    a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """
    係数テーブル a から時刻 t に対応する値を取り出し、x_shape にブロードキャスト可能な形状に変形する。

    Parameters:
    ----------
    a : torch.Tensor
        拡散ステップごとの係数テーブル。shape: (num_timesteps,)。
    t : torch.Tensor
        拡散ステップのインデックス (long 型)。shape: (b,)。
    x_shape : torch.Size | tuple[int, ...]
        ブロードキャスト対象のテンソル形状。

    Returns:
    ----------
    torch.Tensor
        取り出した係数。shape: (b, 1, ..., 1) (次元数は len(x_shape))。
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(
    shape: torch.Size | tuple[int, ...], device: torch.device, repeat: bool = False
) -> torch.Tensor:
    """
    指定形状の標準正規ノイズを生成する。

    Parameters:
    ----------
    shape : torch.Size | tuple[int, ...]
        生成するノイズの形状。
    device : torch.device
        ノイズを生成するデバイス。
    repeat : bool, optional, default=False
        True の場合、バッチ方向 (先頭次元) に同一のノイズを繰り返す。

    Returns:
    ----------
    torch.Tensor
        生成されたノイズ。shape: shape。
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    コサインスケジュールに基づく beta 系列を生成する。
    https://openreview.net/forum?id=-NEXDKk8gZ で提案された手法。

    Parameters:
    ----------
    timesteps : int
        拡散ステップ数。
    s : float, optional, default=0.008
        t = 0 付近で beta が小さくなりすぎないためのオフセット。

    Returns:
    ----------
    np.ndarray
        [0, 0.999] にクリップされた beta 系列。shape: (timesteps,)。
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class ResidualBlock(nn.Module):
    """
    TimeGrad のノイズ予測ネットワークで用いる、膨張畳み込みとゲート付き活性化を持つ残差ブロック。

    Attributes:
    ----------
    target_dim : int
        ターゲット変数の次元数。
    diffusion_projection : nn.Linear
        拡散ステップ埋め込みを residual_channels 次元に射影する線形層。
    dilated_conv : nn.Conv1d
        膨張畳み込み層 (residual_channels -> 2 * residual_channels)。
    conditioner_projection : nn.Conv1d
        条件ベクトルを 2 * residual_channels 次元に射影する畳み込み層。
    output_projection : nn.Conv1d
        出力を residual と skip に射影する 1x1 畳み込み層。
    """

    target_dim: int
    diffusion_projection: nn.Linear
    dilated_conv: nn.Conv1d
    conditioner_projection: nn.Conv1d
    output_projection: nn.Conv1d

    def __init__(
        self, hidden_size: int, residual_channels: int, dilation: int, target_dim: int
    ) -> None:
        """
        ResidualBlock を初期化する。

        Parameters:
        ----------
        hidden_size : int
            拡散ステップ埋め込みの次元数。
        residual_channels : int
            残差チャネル数。
        dilation : int
            膨張畳み込みの膨張率 (target_dim > 1 の場合に使用)。
        target_dim : int
            ターゲット変数の次元数。
        """
        super().__init__()
        self.target_dim = target_dim

        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)

        if self.target_dim > 1:
            self.dilated_conv = nn.Conv1d(
                residual_channels,
                2 * residual_channels,
                3,
                padding=dilation,
                dilation=dilation,
                padding_mode="circular",
            )
            self.conditioner_projection = nn.Conv1d(
                1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
            )
        else:
            self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
            self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1)

        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(
        self, x: torch.Tensor, conditioner: torch.Tensor, diffusion_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        残差ブロックを適用する。

        Parameters:
        ----------
        x : torch.Tensor
            入力。shape: (N, residual_channels, D) (N = B * T, D はパディング後の target_dim)。
        conditioner : torch.Tensor
            アップサンプル済みの条件ベクトル。shape: (N, 1, target_dim)。
        diffusion_step : torch.Tensor
            拡散ステップ埋め込み。shape: (N, hidden_size)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            残差接続後の出力 (shape: (N, residual_channels, D)) とスキップ接続出力 (shape: (N, residual_channels, D))。
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    """
    条件ベクトルを target_dim 次元にアップサンプルする MLP。

    Attributes:
    ----------
    target_dim : int
        ターゲット変数の次元数。
    linear1 : nn.Linear
        cond_length -> target_dim // 2 の線形層 (target_dim > 1 の場合のみ設定)。
    linear2 : nn.Linear
        target_dim // 2 -> target_dim の線形層 (target_dim > 1 の場合のみ設定)。
    linear : nn.Linear
        cond_length -> target_dim の線形層 (target_dim == 1 の場合のみ設定)。
    """

    target_dim: int
    linear1: nn.Linear
    linear2: nn.Linear
    linear: nn.Linear

    def __init__(self, cond_length: int, target_dim: int) -> None:
        """
        CondUpsampler を初期化する。

        Parameters:
        ----------
        cond_length : int
            条件ベクトルの次元数。
        target_dim : int
            ターゲット変数の次元数。
        """
        super().__init__()
        self.target_dim = target_dim

        if self.target_dim > 1:
            self.linear1 = nn.Linear(cond_length, target_dim // 2)
            self.linear2 = nn.Linear(target_dim // 2, target_dim)
        else:
            self.linear = nn.Linear(cond_length, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        条件ベクトルをアップサンプルする。

        Parameters:
        ----------
        x : torch.Tensor
            条件ベクトル。shape: (..., cond_length)。

        Returns:
        ----------
        torch.Tensor
            アップサンプル後の条件ベクトル。shape: (..., target_dim)。
        """
        if self.target_dim > 1:
            x = self.linear1(x)
            x = F.leaky_relu(x, 0.4)
            x = self.linear2(x)
            x = F.leaky_relu(x, 0.4)
        else:
            x = self.linear(x)
            x = F.leaky_relu(x, 0.4)
        return x


class EpsilonTheta(nn.Module):
    """
    TimeGrad のノイズ予測ネットワーク epsilon_theta。

    Attributes:
    ----------
    input_projection : nn.Conv1d
        入力を residual_channels 次元に射影する畳み込み層。
    skip_projection : nn.Conv1d
        スキップ接続の総和に適用する畳み込み層。
    output_projection : nn.Conv1d
        最終出力 (1 チャネル) を得る畳み込み層。重みはゼロ初期化。
    diffusion_embedding : DiffusionEmbedding
        拡散ステップ埋め込み層。
    cond_upsampler : CondUpsampler
        条件ベクトルのアップサンプラ。
    residual_layers : nn.ModuleList
        ResidualBlock のリスト。
    """

    input_projection: nn.Conv1d
    skip_projection: nn.Conv1d
    output_projection: nn.Conv1d
    diffusion_embedding: DiffusionEmbedding
    cond_upsampler: CondUpsampler
    residual_layers: nn.ModuleList

    def __init__(
        self,
        target_dim: int,
        cond_length: int,
        time_emb_dim: int = 16,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
        padding: int = 2,
    ) -> None:
        """
        EpsilonTheta を初期化する。

        Parameters:
        ----------
        target_dim : int
            ターゲット変数の次元数。
        cond_length : int
            条件ベクトルの次元数。
        time_emb_dim : int, optional, default=16
            拡散ステップ正弦波埋め込みの周波数の数。
        residual_layers : int, optional, default=8
            残差ブロックの数。
        residual_channels : int, optional, default=8
            残差チャネル数。
        dilation_cycle_length : int, optional, default=2
            膨張率 2 ** (i % dilation_cycle_length) の周期。
        residual_hidden : int, optional, default=64
            拡散ステップ埋め込みの出力次元数。
        padding : int, optional, default=2
            入力射影の循環パディング幅 (target_dim > 1 の場合に使用)。
        """
        super().__init__()
        if target_dim > 1:
            self.input_projection = nn.Conv1d(
                1, residual_channels, 1, padding=padding, padding_mode="circular"
            )
            self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
            self.output_projection = nn.Conv1d(residual_channels, 1, 3)
        else:
            # self.input_projection = nn.Identity()
            self.input_projection = nn.Conv1d(1, residual_channels, 1)
            self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
            self.output_projection = nn.Conv1d(residual_channels, 1, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                    target_dim=target_dim,
                )
                for i in range(residual_layers)
            ]
        )

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self, inputs: torch.Tensor, time: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        ノイズ付き入力に含まれるノイズを予測する。

        Parameters:
        ----------
        inputs : torch.Tensor
            ノイズ付き入力。shape: (N, 1, target_dim) (N = B * T)。
        time : torch.Tensor
            拡散ステップのインデックス (long 型)。shape: (N,)。
        cond : torch.Tensor
            条件ベクトル。shape: (N, 1, cond_length)。

        Returns:
        ----------
        torch.Tensor
            予測されたノイズ。shape: (N, 1, target_dim)。
        """
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x


class GaussianDiffusion(nn.Module):
    """
    TimeGrad で用いるガウス拡散 (DDPM) モデル。

    Attributes:
    ----------
    dist_args : nn.Linear
        エンコーダの隠れ状態 (f_hidden_size) を条件ベクトル (conditional_length) に射影する線形層。
    denoise_fn : EpsilonTheta
        ノイズ予測ネットワーク。
    target_dim : int
        ターゲット変数の次元数。
    __scale : torch.Tensor | None
        入力のスケーリング係数 (``scale`` プロパティ経由でアクセス)。
    num_timesteps : int
        拡散ステップ数。
    loss_type : str
        損失関数の種類 ("l1", "l2", "huber")。
    betas : torch.Tensor
        beta 系列。shape: (num_timesteps,)。register_buffer で登録。
    alphas_cumprod : torch.Tensor
        alpha の累積積。shape: (num_timesteps,)。register_buffer で登録。
    alphas_cumprod_prev : torch.Tensor
        1 ステップ前の alpha の累積積。shape: (num_timesteps,)。register_buffer で登録。
    sqrt_alphas_cumprod : torch.Tensor
        sqrt(alphas_cumprod)。shape: (num_timesteps,)。register_buffer で登録。
    sqrt_one_minus_alphas_cumprod : torch.Tensor
        sqrt(1 - alphas_cumprod)。shape: (num_timesteps,)。register_buffer で登録。
    log_one_minus_alphas_cumprod : torch.Tensor
        log(1 - alphas_cumprod)。shape: (num_timesteps,)。register_buffer で登録。
    sqrt_recip_alphas_cumprod : torch.Tensor
        sqrt(1 / alphas_cumprod)。shape: (num_timesteps,)。register_buffer で登録。
    sqrt_recipm1_alphas_cumprod : torch.Tensor
        sqrt(1 / alphas_cumprod - 1)。shape: (num_timesteps,)。register_buffer で登録。
    posterior_variance : torch.Tensor
        事後分布 q(x_{t-1} | x_t, x_0) の分散。shape: (num_timesteps,)。register_buffer で登録。
    posterior_log_variance_clipped : torch.Tensor
        クリップした事後分布の対数分散。shape: (num_timesteps,)。register_buffer で登録。
    posterior_mean_coef1 : torch.Tensor
        事後分布の平均における x_0 の係数。shape: (num_timesteps,)。register_buffer で登録。
    posterior_mean_coef2 : torch.Tensor
        事後分布の平均における x_t の係数。shape: (num_timesteps,)。register_buffer で登録。
    """

    dist_args: nn.Linear
    denoise_fn: EpsilonTheta
    target_dim: int
    __scale: torch.Tensor | None
    num_timesteps: int
    loss_type: str
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    def __init__(
        self,
        target_dim: int,
        f_hidden_size: int,
        conditional_length: int,
        beta_end: float = 0.1,
        diff_steps: int = 100,
        loss_type: str = "l2",
        betas: torch.Tensor | np.ndarray | None = None,
        beta_schedule: str = "linear",
        padding: int = 2,
        residual_channels: int = 8,
    ) -> None:
        """
        GaussianDiffusion を初期化し、拡散過程の各種係数を事前計算する。

        Parameters:
        ----------
        target_dim : int
            ターゲット変数の次元数。
        f_hidden_size : int
            エンコーダ出力 (隠れ状態) の次元数。
        conditional_length : int
            条件ベクトルの次元数。
        beta_end : float, optional, default=0.1
            beta スケジュールの終端値。
        diff_steps : int, optional, default=100
            拡散ステップ数。
        loss_type : str, optional, default="l2"
            損失関数の種類 ("l1", "l2", "huber")。
        betas : torch.Tensor | np.ndarray | None, optional, default=None
            beta 系列。指定された場合は beta_schedule より優先される。shape: (diff_steps,)。
        beta_schedule : str, optional, default="linear"
            beta スケジュールの種類 ("linear", "quad", "const", "jsd", "sigmoid", "cosine")。
        padding : int, optional, default=2
            ノイズ予測ネットワークの入力パディング幅。
        residual_channels : int, optional, default=8
            ノイズ予測ネットワークの残差チャネル数。
        """
        super().__init__()
        self.dist_args = nn.Linear(
            in_features=f_hidden_size, out_features=conditional_length
        )
        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditional_length,
            residual_channels=residual_channels,
            padding=padding,
        )
        self.target_dim = target_dim
        self.__scale = None

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4**0.5, beta_end**0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self) -> torch.Tensor | None:
        """
        入力のスケーリング係数を返す。

        Returns:
        ----------
        torch.Tensor | None
            スケーリング係数。x にブロードキャスト可能な形状。
        """
        return self.__scale

    @scale.setter
    def scale(self, scale: torch.Tensor | None) -> None:
        """
        入力のスケーリング係数を設定する。

        Parameters:
        ----------
        scale : torch.Tensor | None
            スケーリング係数。None の場合はスケーリングしない。
        """
        self.__scale = scale

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        拡散過程 q(x_t | x_0) の平均・分散・対数分散を計算する。

        Parameters:
        ----------
        x_start : torch.Tensor
            元データ x_0。shape: (N, 1, target_dim)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            平均 (shape: x_start と同じ)、分散および対数分散 (shape: (N, 1, 1))。
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t と予測ノイズから元データ x_0 を推定する。

        Parameters:
        ----------
        x_t : torch.Tensor
            時刻 t のノイズ付きデータ。shape: (N, 1, target_dim)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。
        noise : torch.Tensor
            予測ノイズ。shape: x_t と同じ。

        Returns:
        ----------
        torch.Tensor
            推定された x_0。shape: x_t と同じ。
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        事後分布 q(x_{t-1} | x_t, x_0) の平均・分散・クリップ済み対数分散を計算する。

        Parameters:
        ----------
        x_start : torch.Tensor
            元データ x_0 (またはその推定値)。shape: (N, 1, target_dim)。
        x_t : torch.Tensor
            時刻 t のノイズ付きデータ。shape: x_start と同じ。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            事後平均 (shape: x_t と同じ)、事後分散およびクリップ済み対数分散 (shape: (N, 1, 1))。
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, clip_denoised: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        逆過程 p(x_{t-1} | x_t) の平均・分散・対数分散を計算する。

        Parameters:
        ----------
        x : torch.Tensor
            時刻 t のノイズ付きデータ。shape: (N, 1, target_dim)。
        cond : torch.Tensor
            条件ベクトル。shape: (N, 1, conditional_length)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。
        clip_denoised : bool
            推定した x_0 を [-1, 1] にクリップするかどうか。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            モデル平均 (shape: x と同じ)、事後分散および事後対数分散 (shape: (N, 1, 1))。
        """
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, cond=cond)
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False,
        repeat_noise: bool = False,
    ) -> torch.Tensor:
        """
        逆過程により x_t から x_{t-1} を 1 ステップサンプリングする。

        Parameters:
        ----------
        x : torch.Tensor
            時刻 t のノイズ付きデータ。shape: (N, 1, target_dim)。
        cond : torch.Tensor
            条件ベクトル。shape: (N, 1, conditional_length)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。
        clip_denoised : bool, optional, default=False
            推定した x_0 を [-1, 1] にクリップするかどうか。
        repeat_noise : bool, optional, default=False
            バッチ方向に同一のノイズを用いるかどうか。

        Returns:
        ----------
        torch.Tensor
            サンプリングされた x_{t-1}。shape: x と同じ。t == 0 の場合はノイズを加えない。
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape: torch.Size | tuple[int, ...], cond: torch.Tensor | None
    ) -> torch.Tensor:
        """
        純粋なノイズから逆過程を全ステップ繰り返してサンプルを生成する。

        Parameters:
        ----------
        shape : torch.Size | tuple[int, ...]
            生成するサンプルの形状。例: (N, 1, target_dim)。
        cond : torch.Tensor | None
            条件ベクトル。shape: (N, 1, conditional_length)。

        Returns:
        ----------
        torch.Tensor
            生成されたサンプル x_0。shape: shape。
        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                img, cond, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        条件ベクトルに基づいてサンプルを生成し、scale が設定されていれば元のスケールに戻す。

        Parameters:
        ----------
        sample_shape : torch.Size, optional, default=torch.Size()
            cond が None の場合に用いるサンプル形状。
        cond : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (N, 1, conditional_length)。指定された場合は cond.shape[:-1] + (target_dim,) をサンプル形状とする。

        Returns:
        ----------
        torch.Tensor
            生成されたサンプル。shape: (N, 1, target_dim)。
        """
        if cond is not None:
            shape = cond.shape[:-1] + (self.target_dim,)
            # TODO reshape cond to (B*T, 1, -1)
        else:
            shape = sample_shape
        x_hat = self.p_sample_loop(shape, cond)  # TODO reshape x_hat to (B,T,-1)

        if self.scale is not None:
            x_hat *= self.scale
        return x_hat

    @torch.no_grad()
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        t: int | None = None,
        lam: float = 0.5,
    ) -> torch.Tensor:
        """
        2 つのデータを時刻 t まで拡散させて潜在空間で線形補間し、逆過程で復元する。

        Parameters:
        ----------
        x1 : torch.Tensor
            1 つ目のデータ。shape: (N, 1, target_dim)。
        x2 : torch.Tensor
            2 つ目のデータ。shape: x1 と同じ。
        t : int | None, optional, default=None
            補間を行う拡散ステップ。None の場合は num_timesteps - 1。
        lam : float, optional, default=0.5
            補間係数 (0 で x1, 1 で x2)。

        Returns:
        ----------
        torch.Tensor
            補間されたサンプル。shape: x1 と同じ。
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        拡散過程 q(x_t | x_0) から x_t をサンプリングする。

        Parameters:
        ----------
        x_start : torch.Tensor
            元データ x_0。shape: (N, 1, target_dim)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。
        noise : torch.Tensor | None, optional, default=None
            加えるノイズ。None の場合は標準正規ノイズを生成する。shape: x_start と同じ。

        Returns:
        ----------
        torch.Tensor
            ノイズ付きデータ x_t。shape: x_start と同じ。
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ノイズ予測の損失を計算する。

        Parameters:
        ----------
        x_start : torch.Tensor
            元データ x_0。shape: (N, 1, target_dim)。
        cond : torch.Tensor
            条件ベクトル。shape: (N, 1, conditional_length)。
        t : torch.Tensor
            拡散ステップのインデックス。shape: (N,)。
        noise : torch.Tensor | None, optional, default=None
            加えるノイズ。None の場合は標準正規ノイズを生成する。shape: x_start と同じ。

        Returns:
        ----------
        torch.Tensor
            損失 (スカラー)。loss_type に応じて L1 / L2 / Huber 損失。
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def loss(
        self, x: torch.Tensor, cond: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        ランダムな拡散ステップをサンプリングして学習損失を計算する。

        Parameters:
        ----------
        x : torch.Tensor
            ターゲットデータ。shape: (B, T, target_dim)。scale が設定されている場合はインプレースで除算される。
        cond : torch.Tensor
            条件ベクトル。shape: (B, T, conditional_length)。
        *args : Any
            p_losses に渡す追加の位置引数。
        **kwargs : Any
            p_losses に渡す追加のキーワード引数。

        Returns:
        ----------
        torch.Tensor
            損失 (スカラー)。
        """
        if self.scale is not None:
            x /= self.scale

        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        loss = self.p_losses(
            x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
        )

        return loss
