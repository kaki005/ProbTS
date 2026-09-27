import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn


class Time_Encoder(nn.Module):
    """
    線形成分と周期 (sin) 成分からなる学習可能な時間エンコーダ。

    Attributes:
    ----------
    periodic : nn.Linear
        周期成分を生成する線形層 (出力次元 embed_time - 1)。
    linear : nn.Linear
        線形成分を生成する線形層 (出力次元 1)。
    """

    periodic: nn.Linear
    linear: nn.Linear

    def __init__(self, embed_time: int) -> None:
        """
        Time_Encoder を初期化する。

        Parameters:
        ----------
        embed_time : int
            時間埋め込みの次元数。
        """
        super().__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        """
        時刻 (位置) を時間埋め込みに変換する。

        Parameters:
        ----------
        tt : torch.Tensor
            時刻テンソル。形状 [B, L, K] または [B, L]。

        Returns:
        ----------
        torch.Tensor
            時間埋め込み。形状 [B, L, K, D] ([B, L] 入力時は [B, L, 1, D])。
        """
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, "b l k -> b l k 1")
        else:  # [B,L]
            tt = rearrange(tt, "b l -> b l 1 1")

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        return out


def sin_cos_encoding(B: int, K: int, L: int, embed_dim: int) -> torch.Tensor:
    """
    sin/cos による絶対位置埋め込みを生成し、バッチ・変数方向に複製する。

    Parameters:
    ----------
    B : int
        バッチサイズ。
    K : int
        変数 (チャネル) 数。
    L : int
        系列長 (位置の数)。
    embed_dim : int
        埋め込み次元数 (偶数である必要がある)。

    Returns:
    ----------
    torch.Tensor
        位置埋め込み。形状 [B, K, L, embed_dim], dtype は torch.float64。
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = [i for i in range(L)]
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    emb = repeat(emb, "l d -> b k l d", b=B, k=K)
    return torch.tensor(emb, dtype=torch.float64)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    与えられた位置のグリッドから 1 次元 sin/cos 位置埋め込みを計算する。

    Parameters:
    ----------
    embed_dim : int
        各位置の出力次元数 (偶数である必要がある)。
    pos : np.ndarray
        エンコードする位置のリスト。サイズ (M,)。

    Returns:
    ----------
    np.ndarray
        位置埋め込み。形状 (M, D)。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
