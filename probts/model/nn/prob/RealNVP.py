# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import copy

import torch
from torch import nn

from probts.model.nn.prob.flow_model import BatchNorm, FlowModel, FlowSequential


class LinearMaskedCoupling(nn.Module):
    """
    MAF 論文に従って修正した RealNVP のカップリング層。

    Attributes:
    ----------
    mask : torch.Tensor
        変換しない次元を 1 とするバイナリマスク。shape: (input_size,)。register_buffer で登録。
    s_net : nn.Sequential
        スケール関数を計算するネットワーク (活性化関数は Tanh)。
    t_net : nn.Sequential
        平行移動関数を計算するネットワーク (活性化関数は ReLU)。
    """

    mask: torch.Tensor
    s_net: nn.Sequential
    t_net: nn.Sequential

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        mask: torch.Tensor,
        cond_label_size: int | None = None,
    ) -> None:
        """
        LinearMaskedCoupling を初期化する。

        Parameters:
        ----------
        input_size : int
            入力の次元数。
        hidden_size : int
            隠れ層の次元数。
        n_hidden : int
            隠れ層の数。
        mask : torch.Tensor
            変換しない次元を 1 とするバイナリマスク。shape: (input_size,)。
        cond_label_size : int | None, optional, default=None
            条件ベクトルの次元数。None の場合は条件なし。
        """
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        データ x を潜在変数 u へ変換する (RealNVP 論文 式 8 参照)。

        Parameters:
        ----------
        x : torch.Tensor
            入力データ。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            潜在変数 u (shape: (..., input_size)) と対数ヤコビアン行列式 log|du/dx| (shape: (..., input_size))。
            input_size 方向の総和はモデルの log_prob 側で取る。
        """
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask
        )

        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        # u = (x - t) * torch.exp(log_s)
        # u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        # log_abs_det_jacobian = -(1 - self.mask) * s
        # log_abs_det_jacobian = -log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = log_s

        return u, log_abs_det_jacobian

    def inverse(
        self, u: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        潜在変数 u をデータ x へ逆変換する。

        Parameters:
        ----------
        u : torch.Tensor
            潜在変数。shape: (..., input_size)。
        y : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (..., cond_label_size)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            データ x (shape: (..., input_size)) と対数ヤコビアン行列式 log|dx/du| (shape: (..., input_size))。
        """
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask
        )

        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        # x = u * torch.exp(log_s) + t
        # x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        # log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        # log_abs_det_jacobian = log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = -log_s

        return x, log_abs_det_jacobian


class RealNVP(FlowModel):
    """
    RealNVP による条件付き正規化フローモデル。

    Attributes:
    ----------
    net : FlowSequential
        カップリング層 (と BatchNorm) を積み重ねたフロー層の列。
    """

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
        batch_norm: bool = True,
    ) -> None:
        """
        RealNVP を初期化する。

        Parameters:
        ----------
        n_blocks : int
            カップリング層ブロックの数。
        target_dim : int
            ターゲット変数の次元数。
        hidden_size : int
            カップリング層の隠れ層の次元数。
        n_hidden : int
            カップリング層の隠れ層の数。
        f_hidden_size : int
            エンコーダ出力 (隠れ状態) の次元数。
        conditional_length : int
            条件ベクトルの次元数。
        dequantize : bool
            対数尤度計算時に一様ノイズを加えて逆量子化するかどうか。
        batch_norm : bool, optional, default=True
            各カップリング層の後に BatchNorm を挿入するかどうか。
        """
        super().__init__(target_dim, f_hidden_size, conditional_length, dequantize)

        # construct model
        modules = []
        mask = torch.arange(target_dim).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    target_dim, hidden_size, n_hidden, mask, conditional_length
                )
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(target_dim)]

        self.net = FlowSequential(*modules)
