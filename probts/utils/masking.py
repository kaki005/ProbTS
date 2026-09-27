# Code implementation from https://github.com/thuml/iTransformer
import torch


class TriangularCausalMask:
    """
    自己注意機構で未来の時刻を参照しないようにするための上三角 causal mask。

    Attributes:
    ----------
    _mask : torch.Tensor
        形状 [B, 1, L, L] の bool マスク (True の位置がマスクされる)。
    """

    _mask: torch.Tensor

    def __init__(self, B: int, L: int, device: str | torch.device = "cpu") -> None:
        """
        TriangularCausalMask を初期化する。

        Parameters:
        ----------
        B : int
            バッチサイズ。
        L : int
            系列長。
        device : str | torch.device, optional, default="cpu"
            マスクを配置するデバイス。
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        causal mask を返す。

        Returns:
        ----------
        torch.Tensor
            形状 [B, 1, L, L] の bool マスク。
        """
        return self._mask


class ProbMask:
    """
    ProbSparse attention (Informer) 向けに、選択されたクエリに対応する causal mask を生成するクラス。

    Attributes:
    ----------
    _mask : torch.Tensor
        scores と同じ形状の bool マスク (True の位置がマスクされる)。
    """

    _mask: torch.Tensor

    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: torch.Tensor,
        scores: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        ProbMask を初期化する。

        Parameters:
        ----------
        B : int
            バッチサイズ。
        H : int
            アテンションヘッド数。
        L : int
            クエリの系列長。
        index : torch.Tensor
            選択された上位クエリのインデックス。形状 [B, H, n_top]。
        scores : torch.Tensor
            アテンションスコア。形状 [B, H, n_top, L_K]。
        device : str | torch.device, optional, default="cpu"
            マスクを配置するデバイス。
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """
        ProbSparse attention 用のマスクを返す。

        Returns:
        ----------
        torch.Tensor
            scores と同じ形状の bool マスク。
        """
        return self._mask
