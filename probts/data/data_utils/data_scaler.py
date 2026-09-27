# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
from torch import nn


class Scaler:
    """
    データのスケーリング (正規化) を行うスケーラーの基底クラス。

    サブクラスは fit / transform / fit_transform / inverse_transform を実装する必要がある。
    """

    def __init__(self) -> None:
        """
        Scaler を初期化する。
        """
        super().__init__()

    def fit(self, values: torch.Tensor) -> None:
        """
        入力データからスケーリングに用いる統計量を計算する。

        Parameters:
        ----------
        values : torch.Tensor
            統計量の計算に用いる入力テンソル。

        Raises:
        ----------
        NotImplementedError
            サブクラスで実装されていない場合。
        """
        raise NotImplementedError

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        計算済みの統計量を用いて入力データをスケーリングする。

        Parameters:
        ----------
        values : torch.Tensor
            スケーリング対象の入力テンソル。

        Returns:
        ----------
        torch.Tensor
            スケーリング後のテンソル。

        Raises:
        ----------
        NotImplementedError
            サブクラスで実装されていない場合。
        """
        raise NotImplementedError

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        統計量の計算とスケーリングを続けて行う。

        Parameters:
        ----------
        values : torch.Tensor
            入力テンソル。

        Returns:
        ----------
        torch.Tensor
            スケーリング後のテンソル。

        Raises:
        ----------
        NotImplementedError
            サブクラスで実装されていない場合。
        """
        raise NotImplementedError

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        スケーリングされたデータを元のスケールに戻す。

        Parameters:
        ----------
        values : torch.Tensor
            スケーリング済みのテンソル。

        Returns:
        ----------
        torch.Tensor
            元のスケールに戻したテンソル。

        Raises:
        ----------
        NotImplementedError
            サブクラスで実装されていない場合。
        """
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    PyTorch のネイティブ関数を用いてテンソルを標準化 (平均 0, 標準偏差 1) するスケーラー。

    テンソルの形状は特に問わず、特徴量 (変量) が最後の次元にあれば動作する。

    Attributes:
    ----------
    mean : torch.Tensor | float | None
        特徴量の平均。fit 呼び出し後に設定される。
    scale : torch.Tensor | float | None
        特徴量の標準偏差。fit 呼び出し後に設定される。
    epsilon : float
        ゼロ除算を避けるための微小値。
    var_specific : bool
        True の場合、平均と標準偏差を変量ごとに計算する。
    """

    mean: torch.Tensor | float | None
    scale: torch.Tensor | float | None
    epsilon: float
    var_specific: bool

    def __init__(
        self,
        mean: torch.Tensor | float | None = None,
        std: torch.Tensor | float | None = None,
        epsilon: float = 1e-9,
        var_specific: bool = True,
    ) -> None:
        """
        StandardScaler を初期化する。

        Parameters:
        ----------
        mean : torch.Tensor | float | None, optional, default=None
            特徴量の平均。fit 呼び出し後に設定される。
        std : torch.Tensor | float | None, optional, default=None
            特徴量の標準偏差。fit 呼び出し後に設定される。
        epsilon : float, optional, default=1e-9
            ゼロ除算 (Division-By-Zero) を避けるための微小値。
        var_specific : bool, optional, default=True
            True の場合、平均と標準偏差を変量ごとに計算する。
        """
        self.mean = mean
        self.scale = std
        self.epsilon = epsilon
        self.var_specific = var_specific

    def fit(self, values: torch.Tensor) -> None:
        """
        入力データから平均と標準偏差を計算し、属性に設定する。

        Parameters:
        ----------
        values : torch.Tensor
            形状 (T, C) または (N, T, C) の入力テンソル。
            N はバッチサイズ、T は時間ステップ数、C は変量数。
        """
        dims = list(range(values.dim() - 1))
        if not self.var_specific:
            self.mean = torch.mean(values)
            self.scale = torch.std(values)
        else:
            self.mean = torch.mean(values, dim=dims)
            self.scale = torch.std(values, dim=dims)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        計算済みの平均と標準偏差を用いて入力データを標準化する。

        Parameters:
        ----------
        values : torch.Tensor
            標準化対象の入力テンソル。

        Returns:
        ----------
        torch.Tensor
            標準化後の float32 テンソル。mean が未設定の場合は入力をそのまま返す。
        """
        if self.mean is None:
            return values

        values = (values - self.mean.to(values.device)) / (
            self.scale.to(values.device) + self.epsilon
        )
        return values.to(torch.float32)

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        平均と標準偏差を計算した後、入力データを標準化する。

        Parameters:
        ----------
        values : torch.Tensor
            形状 (T, C) または (N, T, C) の入力テンソル。

        Returns:
        ----------
        torch.Tensor
            標準化後のテンソル。
        """
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        標準化されたデータを元のスケールに戻す。

        Parameters:
        ----------
        values : torch.Tensor
            標準化済みのテンソル。

        Returns:
        ----------
        torch.Tensor
            元のスケールに戻したテンソル。mean が未設定の場合は入力をそのまま返す。
        """
        if self.mean is None:
            return values

        values = values * (self.scale.to(values.device) + self.epsilon)
        values = values + self.mean.to(values.device)
        return values


class TemporalScaler(Scaler):
    """
    各アイテムについて、時間方向の絶対値平均に基づくスケールを計算するスケーラー。

    平均は観測インジケータで示された観測済みの値のみから計算する。
    観測値を持たないアイテムには、全体平均に基づくスケールが割り当てられる。

    Attributes:
    ----------
    scale : torch.Tensor | None
        計算されたスケール。形状は (N, 1, C) または (N, C, 1)。fit 呼び出し後に設定される。
    minimum_scale : torch.Tensor
        時系列がゼロのみの場合などに用いる最小スケール。
    time_first : bool
        True の場合、入力テンソルの形状は (N, T, C)、False の場合は (N, C, T)。
    """

    scale: torch.Tensor | None
    minimum_scale: torch.Tensor
    time_first: bool

    def __init__(self, minimum_scale: float = 1e-10, time_first: bool = True) -> None:
        """
        TemporalScaler を初期化する。

        Parameters:
        ----------
        minimum_scale : float, optional, default=1e-10
            時系列がゼロのみの場合に用いるデフォルト (最小) スケール。
        time_first : bool, optional, default=True
            True の場合、入力テンソルの形状は (N, T, C)、False の場合は (N, C, T)。
        """
        super().__init__()
        self.scale = None
        self.minimum_scale = torch.tensor(minimum_scale)
        self.time_first = time_first

    def fit(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = None
    ) -> None:
        """
        データに対してスケーラーを適合させ、スケールを計算して属性に設定する。

        計算されるスケールの形状は (N, 1, C) または (N, C, 1) となる。

        Parameters:
        ----------
        data : torch.Tensor
            スケーリング対象のデータ。``time_first == True`` の場合は形状 (N, T, C)、
            ``time_first == False`` の場合は形状 (N, C, T)。
        observed_indicator : torch.Tensor | None, optional, default=None
            ``data`` と同じ形状のバイナリテンソル。観測済みのデータ点に 1、
            欠損しているデータ点に 0 を持つ。None の場合は全て観測済みとみなす。
        """
        if self.time_first:
            dim = -2
        else:
            dim = -1

        if observed_indicator is None:
            observed_indicator = torch.ones_like(data)

        # These will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # First compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # Then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # Use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        self.scale = torch.max(scale, self.minimum_scale).unsqueeze(dim=dim).detach()

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        計算済みのスケールでデータを割ってスケーリングする。

        Parameters:
        ----------
        data : torch.Tensor
            スケーリング対象のテンソル。

        Returns:
        ----------
        torch.Tensor
            スケーリング後のテンソル。
        """
        return data / self.scale.to(data.device)

    def fit_transform(
        self, data: torch.Tensor, observed_indicator: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        スケールを計算した後、データをスケーリングする。

        Parameters:
        ----------
        data : torch.Tensor
            スケーリング対象のテンソル。
        observed_indicator : torch.Tensor | None, optional, default=None
            ``data`` と同じ形状の観測インジケータ (観測済み 1, 欠損 0)。

        Returns:
        ----------
        torch.Tensor
            スケーリング後のテンソル。
        """
        self.fit(data, observed_indicator)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        スケーリングされたデータに計算済みのスケールを掛けて元のスケールに戻す。

        Parameters:
        ----------
        data : torch.Tensor
            スケーリング済みのテンソル。

        Returns:
        ----------
        torch.Tensor
            元のスケールに戻したテンソル。
        """
        return data * self.scale.to(data.device)


class IdentityScaler(Scaler):
    """
    スケーリングを一切行わない恒等スケーラー。

    Attributes:
    ----------
    scale : None
        常に None (互換性のための属性)。
    """

    scale: None

    def __init__(self, time_first: bool = True) -> None:
        """
        IdentityScaler を初期化する。

        Parameters:
        ----------
        time_first : bool, optional, default=True
            他のスケーラーとのインターフェース互換のための引数 (未使用)。
        """
        super().__init__()
        self.scale = None

    def fit(self, data: torch.Tensor) -> None:
        """
        何も行わない。

        Parameters:
        ----------
        data : torch.Tensor
            入力テンソル (未使用)。
        """
        pass

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        入力データをそのまま返す。

        Parameters:
        ----------
        data : torch.Tensor
            入力テンソル。

        Returns:
        ----------
        torch.Tensor
            入力と同じテンソル。
        """
        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        入力データをそのまま返す。

        Parameters:
        ----------
        data : torch.Tensor
            入力テンソル。

        Returns:
        ----------
        torch.Tensor
            入力と同じテンソル。
        """
        return data


class InstanceNorm(nn.Module):
    """
    インスタンスごとに平均と標準偏差で正規化・逆正規化を行うモジュール。

    Attributes:
    ----------
    eps : float
        数値安定性のために分散に加える微小値。
    mean : torch.Tensor
        正規化時に計算された平均 (mode="norm" で forward 呼び出し後に設定)。
    stdev : torch.Tensor
        正規化時に計算された標準偏差 (mode="norm" で forward 呼び出し後に設定)。
    """

    eps: float
    mean: torch.Tensor
    stdev: torch.Tensor

    def __init__(self, eps: float = 1e-5) -> None:
        """
        InstanceNorm を初期化する。

        Parameters:
        ----------
        eps : float, optional, default=1e-5
            数値安定性のために加える微小値。
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        指定されたモードに応じて正規化または逆正規化を行う。

        Parameters:
        ----------
        x : torch.Tensor
            形状 (B, ..., C) の入力テンソル。
        mode : str
            "norm" の場合は統計量を計算して正規化し、"denorm" の場合は逆正規化する。

        Returns:
        ----------
        torch.Tensor
            正規化または逆正規化後のテンソル。

        Raises:
        ----------
        NotImplementedError
            mode が "norm" / "denorm" 以外の場合。
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x: torch.Tensor) -> None:
        """
        最初と最後を除く次元に沿って平均と標準偏差を計算し、属性に設定する。

        Parameters:
        ----------
        x : torch.Tensor
            入力テンソル。
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算済みの平均と標準偏差で入力を正規化する。

        Parameters:
        ----------
        x : torch.Tensor
            入力テンソル。

        Returns:
        ----------
        torch.Tensor
            正規化後のテンソル。
        """
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算済みの平均と標準偏差を用いて入力を元のスケールに戻す。

        Parameters:
        ----------
        x : torch.Tensor
            正規化済みのテンソル。

        Returns:
        ----------
        torch.Tensor
            逆正規化後のテンソル。
        """
        x = x * self.stdev
        x = x + self.mean
        return x
