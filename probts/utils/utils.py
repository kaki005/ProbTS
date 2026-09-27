# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import importlib
import os
import re

import torch


def repeat(tensor: torch.Tensor, n: int, dim: int = 0) -> torch.Tensor:
    """
    テンソルの各要素を指定した次元方向に n 回繰り返す (repeat_interleave)。

    Parameters:
    ----------
    tensor : torch.Tensor
        繰り返し対象のテンソル。
    n : int
        繰り返し回数。
    dim : int, optional, default=0
        繰り返しを行う次元。

    Returns:
    ----------
    torch.Tensor
        繰り返し後のテンソル。
    """
    return tensor.repeat_interleave(repeats=n, dim=dim)


def extract(
    a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """
    係数テンソル a からタイムステップ t に対応する値を取り出し、x_shape にブロードキャスト可能な形状に変形する。

    拡散モデルにおいて、各サンプルのタイムステップに対応するスケジュール係数を取得するために使用する。

    Parameters:
    ----------
    a : torch.Tensor
        タイムステップごとの係数。形状 (num_timesteps,)。
    t : torch.Tensor
        各サンプルのタイムステップのインデックス。形状 (batch_size,)。
    x_shape : torch.Size | tuple[int, ...]
        ブロードキャスト先となる入力テンソルの形状。

    Returns:
    ----------
    torch.Tensor
        形状 (batch_size, 1, ..., 1) の係数テンソル (t と同じデバイス上)。
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def weighted_average(
    x: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: int | None = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    指定した次元に沿ってテンソルの重み付き平均を計算する。重みが 0 の要素はマスクされる。

    すなわち `nan * 0 = nan` とならず `0 * 0 = 0` となる。

    Parameters:
    ----------
    x : torch.Tensor
        平均を計算する入力テンソル。
    weights : torch.Tensor | None, optional, default=None
        重みテンソル (`x` と同じ形状)。None の場合は単純平均を計算する。
    dim : int | None, optional, default=None
        `x` の平均を取る次元。None (または 0) の場合は全要素で平均する (weights が None の場合は x をそのまま返す)。
    reduce : str, optional, default="mean"
        "mean" の場合は重み付き平均を返し、それ以外の場合は重み付けのみ行ったテンソルを返す。

    Returns:
    ----------
    torch.Tensor
        指定した `dim` に沿って平均されたテンソル (reduce != "mean" の場合は重み付けされたテンソル)。
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != "mean":
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x


def convert_to_list(s: str | list[int] | int | None) -> list[int] | None:
    """
    予測長の文字列などをリストに変換する。

    例: '96-192-336-720' は [96, 192, 336, 720] に変換される。

    Parameters:
    ----------
    s : str | list[int] | int | None
        変換対象の値 (str, list, int)。

    Returns:
    ----------
    list[int] | None
        変換後のリスト。サポート外の型 (None など) の場合は None。
    """
    if type(s).__name__ == "int":
        return [s]
    elif type(s).__name__ == "list":
        return s
    elif type(s).__name__ == "str":
        elements = re.split(r"\D+", s)
        return list(map(int, elements))
    else:
        return None


def find_best_epoch(ckpt_folder: str) -> tuple[int, str]:
    """
    チェックポイントフォルダ内から val_CRPS が最小のエポックを探す。

    CRPS 値の比較に関する問題を特定・修正してくれた GitHub@Kai-Ref に感謝する。

    Parameters:
    ----------
    ckpt_folder : str
        "epoch={epoch}-val_CRPS={crps}" 形式のチェックポイントファイルを含むフォルダのパス。

    Returns:
    ----------
    tuple[int | None, str | None]
        (最良エポック番号, 最良チェックポイントのファイル名)。該当ファイルがない場合は (None, None)。
    """
    pattern = r"epoch=(\d+)-val_CRPS=([0-9]*\.[0-9]+)"
    ckpt_files = os.listdir(ckpt_folder)  # List of checkpoint files

    best_ckpt = None
    best_epoch = None
    best_crps = float("inf")  # Start with an infinitely large CRPS

    for filename in ckpt_files:
        match = re.search(pattern, filename)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            crps = float(match.group(2))  # Extract CRPS value

            if crps < best_crps:  # If this is the lowest CRPS found so far
                best_crps = crps
                best_ckpt = filename
                best_epoch = epoch  # Store the best epoch number
    assert best_epoch is not None and best_ckpt is not None
    return best_epoch, best_ckpt


def ensure_list(
    input_value: str | list[int] | int | None,
    default_value: str | list[int] | int | None = None,
) -> list[int] | None:
    """
    入力をリストに変換する。入力が None (変換不可) の場合は、代わりにデフォルト値をリストに変換する。

    Parameters:
    ----------
    input_value : str | list[int] | int | None
        変換対象の値。
    default_value : str | list[int] | int | None, optional, default=None
        input_value が変換できない場合に使用するデフォルト値。

    Returns:
    ----------
    list[int] | None
        変換後のリスト。どちらも変換できない場合は None。
    """
    result = convert_to_list(input_value)
    if result is None:
        result = convert_to_list(default_value)
    return result


def init_class_helper(class_name: str) -> type:
    """
    モジュールを動的に import し、クラスを取得する。

    Parameters:
    ----------
    class_name : str
        "module_name.ClassName" 形式のクラスの完全修飾名。

    Returns:
    ----------
    type
        指定したモジュールから取得したクラスオブジェクト。
    """
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class
