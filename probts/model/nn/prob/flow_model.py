# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
from torch import nn
from torch.distributions import Normal


class FlowModel(nn.Module):
    """
    条件付き正規化フロー (Normalizing Flow) モデルの基底クラス。

    サブクラス (MAF, RealNVP など) が ``net`` にフロー層の列を設定して使用する。

    Attributes:
    ----------
    __scale : torch.Tensor | None
        入力のスケーリング係数 (``scale`` プロパティ経由でアクセス)。None の場合はスケーリングしない。
    net : FlowSequential | None
        フロー層の列。サブクラスで設定される。
    dequantize : bool
        対数尤度計算時に一様ノイズを加えて逆量子化するかどうか。
    dist_args : nn.Linear
        エンコーダの隠れ状態 (f_hidden_size) を条件ベクトル (conditional_length) に射影する線形層。
    base_dist_mean : torch.Tensor
        基底分布 (正規分布) の平均。shape: (target_dim,)。register_buffer で登録。
    base_dist_var : torch.Tensor
        基底分布 (正規分布) のスケール。shape: (target_dim,)。register_buffer で登録。
    """

    __scale: torch.Tensor | None
    net: "FlowSequential | None"
    dequantize: bool
    dist_args: nn.Linear
    base_dist_mean: torch.Tensor
    base_dist_var: torch.Tensor

    def __init__(
        self,
        target_dim: int,
        f_hidden_size: int,
        conditional_length: int,
        dequantize: bool,
    ) -> None:
        """
        FlowModel を初期化する。

        Parameters:
        ----------
        target_dim : int
            ターゲット変数の次元数。
        f_hidden_size : int
            エンコーダ出力 (隠れ状態) の次元数。
        conditional_length : int
            フローに与える条件ベクトルの次元数。
        dequantize : bool
            対数尤度計算時に一様ノイズを加えて逆量子化するかどうか。
        """
        super().__init__()
        self.__scale = None
        self.net = None
        self.dequantize = dequantize

        self.dist_args = nn.Linear(
            in_features=f_hidden_size, out_features=conditional_length
        )

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(target_dim))
        self.register_buffer("base_dist_var", torch.ones(target_dim))

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

    @property
    def scale(self) -> torch.Tensor | None:
        """
        入力のスケーリング係数を返す。

        Returns:
        ----------
        torch.Tensor | None
            スケーリング係数。shape: (B, 1, target_dim) など x にブロードキャスト可能な形状。
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

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        データ x を潜在変数 u へ変換する (順方向フロー)。

        Parameters:
        ----------
        x : torch.Tensor
            入力データ。shape: (B, T, target_dim)。scale が設定されている場合はインプレースで除算される。
        cond : torch.Tensor
            条件ベクトル。shape: (B, T, conditional_length)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            潜在変数 u (shape: (B, T, target_dim)) と対数ヤコビアン行列式の総和 (shape: (B, T, target_dim))。
        """
        if self.scale is not None:
            x /= self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        return u, log_abs_det_jacobian

    def inverse(
        self, u: torch.Tensor, cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        潜在変数 u をデータ空間 x へ変換する (逆方向フロー)。

        Parameters:
        ----------
        u : torch.Tensor
            潜在変数。shape: (B, T, target_dim)。
        cond : torch.Tensor | None
            条件ベクトル。shape: (B, T, conditional_length)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            データ空間のサンプル x (shape: (B, T, target_dim)) と対数ヤコビアン行列式の総和。
        """
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x *= self.scale
            log_abs_det_jacobian += torch.log(torch.abs(self.scale))
        return x, log_abs_det_jacobian

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        条件付き対数尤度を計算する。

        Parameters:
        ----------
        x : torch.Tensor
            観測データ。shape: (B, T, target_dim)。
        cond : torch.Tensor
            条件ベクトル。shape: (B, T, conditional_length)。

        Returns:
        ----------
        torch.Tensor
            対数尤度。shape: (B, T)。
        """
        if self.dequantize:
            x += torch.rand_like(x)
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1)

    def loss(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        負の対数尤度を損失として計算する。

        Parameters:
        ----------
        x : torch.Tensor
            観測データ。shape: (B, T, target_dim)。
        cond : torch.Tensor
            条件ベクトル。shape: (B, T, conditional_length)。

        Returns:
        ----------
        torch.Tensor
            負の対数尤度。shape: (B, T)。
        """
        return -self.log_prob(x, cond)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        基底分布からサンプリングし、逆方向フローでデータ空間のサンプルを生成する。

        Parameters:
        ----------
        sample_shape : torch.Size, optional, default=torch.Size()
            cond が None の場合に用いるサンプル形状。
        cond : torch.Tensor | None, optional, default=None
            条件ベクトル。shape: (B, T, conditional_length)。指定された場合は cond.shape[:-1] をサンプル形状とする。

        Returns:
        ----------
        torch.Tensor
            生成されたサンプル。shape: (B, T, target_dim)。
        """
        if cond is not None:
            shape = cond.shape[:-1]
        else:
            shape = sample_shape

        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class BatchNorm(nn.Module):
    """
    フローモデル用の BatchNorm 層 (可逆変換として対数ヤコビアンも返す)。

    Attributes:
    ----------
    momentum : float
        移動平均・移動分散の更新に用いるモメンタム。
    eps : float
        数値安定化のための微小値。
    log_gamma : nn.Parameter
        スケールの対数。shape: (input_size,)。
    beta : nn.Parameter
        シフト量。shape: (input_size,)。
    running_mean : torch.Tensor
        移動平均。shape: (input_size,)。register_buffer で登録。
    running_var : torch.Tensor
        移動分散。shape: (input_size,)。register_buffer で登録。
    batch_mean : torch.Tensor
        直近の学習時バッチの平均。shape: (input_size,)。学習時の forward で設定。
    batch_var : torch.Tensor
        直近の学習時バッチの分散。shape: (input_size,)。学習時の forward で設定。
    """

    momentum: float
    eps: float
    log_gamma: nn.Parameter
    beta: nn.Parameter
    running_mean: torch.Tensor
    running_var: torch.Tensor
    batch_mean: torch.Tensor
    batch_var: torch.Tensor

    def __init__(
        self, input_size: int, momentum: float = 0.9, eps: float = 1e-5
    ) -> None:
        """
        BatchNorm を初期化する。

        Parameters:
        ----------
        input_size : int
            入力の次元数。
        momentum : float, optional, default=0.9
            移動平均・移動分散の更新に用いるモメンタム。
        eps : float, optional, default=1e-5
            数値安定化のための微小値。
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))

    def forward(
        self, x: torch.Tensor, cond_y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        入力を正規化する (順方向)。学習時はバッチ統計量を計算し移動統計量を更新する。

        Parameters:
        ----------
        x : torch.Tensor
            入力。shape: (..., input_size)。
        cond_y : torch.Tensor | None, optional, default=None
            条件ベクトル (未使用。FlowSequential とのインターフェース互換のため)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            正規化後の出力 y (shape: (..., input_size)) と対数ヤコビアン行列式 (shape: x と同じ)。
        """
        if self.training:
            self.batch_mean = x.view(-1, x.shape[-1]).mean(0)
            # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            self.batch_var = x.view(-1, x.shape[-1]).var(0)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum)
            )
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum)
            )

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)

        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(
        self, y: torch.Tensor, cond_y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        正規化を逆変換する (逆方向)。

        Parameters:
        ----------
        y : torch.Tensor
            正規化後の値。shape: (..., input_size)。
        cond_y : torch.Tensor | None, optional, default=None
            条件ベクトル (未使用。FlowSequential とのインターフェース互換のため)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            逆変換後の値 x (shape: (..., input_size)) と対数ヤコビアン行列式 (shape: x と同じ)。
        """
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """
    正規化フローの層を順に適用するコンテナ。

    各層の対数ヤコビアン行列式を累積して返す。
    """

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        各層を順方向に適用する。

        Parameters:
        ----------
        x : torch.Tensor
            入力データ。shape: (..., target_dim)。
        y : torch.Tensor | None
            条件ベクトル。shape: (..., conditional_length)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            変換後の潜在変数 (shape: (..., target_dim)) と対数ヤコビアン行列式の総和。
        """
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(
        self, u: torch.Tensor, y: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        各層を逆順に逆変換として適用する。

        Parameters:
        ----------
        u : torch.Tensor
            潜在変数。shape: (..., target_dim)。
        y : torch.Tensor | None
            条件ベクトル。shape: (..., conditional_length)。

        Returns:
        ----------
        tuple[torch.Tensor, torch.Tensor]
            データ空間の値 (shape: (..., target_dim)) と対数ヤコビアン行列式の総和。
        """
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians
