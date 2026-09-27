# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from probts.model.nn.prob.flow_model import BatchNorm, FlowModel, FlowSequential


def create_masks(
    input_size: int,
    hidden_size: int,
    n_hidden: int,
    input_order: str = "sequential",
    input_degrees: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    MADE の自己回帰性を保証するための各層のマスクを生成する (MADE 論文 4 節)。

    Parameters:
    ----------
    input_size : int
        入力の次元数。
    hidden_size : int
        隠れ層の次元数。
    n_hidden : int
        隠れ層の数。
    input_order : str, optional, default="sequential"
        入力の次数 (degree) の決め方 ("sequential" または "random")。
    input_degrees : torch.Tensor | None, optional, default=None
        入力の次数。MADE を積み重ねる場合に、前段の次数を反転したものを与える。shape: (input_size,)。

    Returns:
    ----------
    tuple[list[torch.Tensor], torch.Tensor]
        各層のマスクのリスト (各要素の shape: (out_features, in_features)) と、入力層の次数 (shape: (input_size,))。
    """
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == "sequential":
        degrees += (
            [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += (
            [torch.arange(input_size) % input_size - 1]
            if input_degrees is None
            else [input_degrees % input_size - 1]
        )

    elif input_order == "random":
        degrees += (
            [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        )
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += (
            [torch.randint(min_prev_degree, input_size, (input_size,)) - 1]
            if input_degrees is None
            else [input_degrees - 1]
        )

    # construct masks
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """
    MADE の構成要素となるマスク付き線形層。

    Attributes:
    ----------
    mask : torch.Tensor
        重みに掛けるマスク。shape: (n_outputs, input_size)。register_buffer で登録。
    cond_label_size : int | None
        条件ベクトルの次元数。None の場合は条件なし。
    cond_weight : nn.Parameter
        条件ベクトルに対する重み。shape: (n_outputs, cond_label_size)。cond_label_size が None でない場合のみ設定。
    """

    mask: torch.Tensor
    cond_label_size: int | None
    cond_weight: nn.Parameter

    def __init__(
        self,
        input_size: int,
        n_outputs: int,
        mask: torch.Tensor,
        cond_label_size: int | None = None,
    ) -> None:
        """
        MaskedLinear を初期化する。

        Parameters:
        ----------
        input_size : int
            入力の次元数。
        n_outputs : int
            出力の次元数。
        mask : torch.Tensor
            重みに掛けるマスク。shape: (n_outputs, input_size)。
        cond_label_size : int | None, optional, default=None
            条件ベクトルの次元数。None の場合は条件なし。
        """
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        マスク付き線形変換を適用し、条件ベクトルがあればその線形項を加える。

        Parameters:
        ----------
        x : torch.Tensor
            入力。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。

        Returns:
        ----------
        torch.Tensor
            出力。shape: (..., n_outputs)。
        """
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class MADE(nn.Module):
    """
    MADE (Masked Autoencoder for Distribution Estimation) による自己回帰変換層。

    Attributes:
    ----------
    base_dist_mean : torch.Tensor
        基底分布 (正規分布) の平均。shape: (input_size,)。register_buffer で登録。
    base_dist_var : torch.Tensor
        基底分布 (正規分布) のスケール。shape: (input_size,)。register_buffer で登録。
    input_degrees : torch.Tensor
        入力層の次数 (変数の順序)。shape: (input_size,)。
    net_input : MaskedLinear
        入力層 (条件ベクトルを受け取るマスク付き線形層)。
    net : nn.Sequential
        隠れ層および出力層 (平均と対数スケールを出力)。
    """

    base_dist_mean: torch.Tensor
    base_dist_var: torch.Tensor
    input_degrees: torch.Tensor
    net_input: MaskedLinear
    net: nn.Sequential

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        cond_label_size: int | None = None,
        activation: str = "ReLU",
        input_order: str = "sequential",
        input_degrees: torch.Tensor | None = None,
    ) -> None:
        """
        MADE を初期化する。

        Parameters:
        ----------
        input_size : int
            入力の次元数。
        hidden_size : int
            隠れ層の次元数。
        n_hidden : int
            隠れ層の数。
        cond_label_size : int | None, optional, default=None
            条件ベクトルの次元数。None の場合は条件なし。
        activation : str, optional, default="ReLU"
            使用する活性化関数 ("ReLU" または "Tanh")。
        input_order : str, optional, default="sequential"
            自己回帰マスク生成時の変数順序 ("sequential" または "random")。
        input_degrees : torch.Tensor | None, optional, default=None
            MADE を積み重ねる場合に前段から反転して引き継ぐ入力の次数。shape: (input_size,)。
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        # setup activation
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(
            input_size, hidden_size, masks[0], cond_label_size
        )
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self) -> Normal:
        """
        基底分布 (標準正規分布) を返す。

        Returns:
        ----------
        Normal
            平均 base_dist_mean, スケール base_dist_var の正規分布。
        """
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        データ x を潜在変数 u へ変換する (MAF 論文 式 4, 5)。

        Parameters:
        ----------
        x : torch.Tensor
            入力データ。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            潜在変数 u (shape: (..., input_size)) と対数ヤコビアン行列式 (shape: (..., input_size))。
        """
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(
        self,
        u: torch.Tensor,
        y: torch.Tensor | None = None,
        sum_log_abs_det_jacobians: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        潜在変数 u をデータ x へ逐次的に逆変換する (MAF 論文 式 3)。

        Parameters:
        ----------
        u : torch.Tensor
            潜在変数。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。
        sum_log_abs_det_jacobians : torch.Tensor | None, optional, default=None
            未使用。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            データ x (shape: (..., input_size)) と対数ヤコビアン行列式 (shape: (..., input_size))。
        """
        # MAF eq 3
        # D = u.shape[-1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            x[..., i] = u[..., i] * torch.exp(loga[..., i]) + m[..., i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        対数尤度を計算する。

        Parameters:
        ----------
        x : torch.Tensor
            入力データ。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。

        Returns:
        ----------
        torch.Tensor
            対数尤度。shape: (...,)。
        """
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class MAF(FlowModel):
    """
    MAF (Masked Autoregressive Flow) による条件付き正規化フローモデル。

    Attributes:
    ----------
    input_degrees : torch.Tensor | None
        最後に追加した MADE ブロックの入力次数を反転したもの。shape: (target_dim,)。
    net : FlowSequential
        MADE ブロック (と BatchNorm) を積み重ねたフロー層の列。
    """

    input_degrees: torch.Tensor | None
    net: FlowSequential

    def __init__(
        self,
        n_blocks: int,
        target_dim: int,
        hidden_size: int,
        n_hidden: int,
        f_hidden_size: int,
        conditional_length: int,
        dequantize: bool,
        activation: str = "ReLU",
        input_order: str = "sequential",
        batch_norm: bool = True,
    ) -> None:
        """
        MAF を初期化する。

        Parameters:
        ----------
        n_blocks : int
            MADE ブロックの数。
        target_dim : int
            ターゲット変数の次元数。
        hidden_size : int
            MADE の隠れ層の次元数。
        n_hidden : int
            MADE の隠れ層の数。
        f_hidden_size : int
            エンコーダ出力 (隠れ状態) の次元数。
        conditional_length : int
            条件ベクトルの次元数。
        dequantize : bool
            対数尤度計算時に一様ノイズを加えて逆量子化するかどうか。
        activation : str, optional, default="ReLU"
            MADE で使用する活性化関数 ("ReLU" または "Tanh")。
        input_order : str, optional, default="sequential"
            自己回帰マスク生成時の変数順序 ("sequential" または "random")。
        batch_norm : bool, optional, default=True
            各 MADE ブロックの後に BatchNorm を挿入するかどうか。
        """
        super().__init__(target_dim, f_hidden_size, conditional_length, dequantize)

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(
                    target_dim,
                    hidden_size,
                    n_hidden,
                    conditional_length,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(target_dim)]

        self.net = FlowSequential(*modules)
