# ---------------------------------------------------------------------------------
# Portions of this file are derived from Chronos
# - Source: https://github.com/amazon-science/chronos-forecasting
# - Paper: Chronos: Learning the Language of Time Series
# - License: Apache License 2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import TYPE_CHECKING, Any

import torch

# from chronos import ChronosPipeline
from einops import rearrange

from probts.model.forecaster import Forecaster
from probts.model.nn.arch.ChronosModule.base import BaseChronosPipeline

if TYPE_CHECKING:
    from probts.data.data_wrapper import ProbTSBatchData


class Chronos(Forecaster):
    """
    事前学習済み Chronos (T5 ベース) を用いたゼロショット確率的予測モデル。

    Attributes:
    ----------
    pred_len : int
        予測長 (prediction_length の最大値)。
    pipeline : BaseChronosPipeline
        事前学習済み Chronos パイプライン。
    q : list[float]
        分位点レベルのリスト。
    """

    pred_len: int
    pipeline: BaseChronosPipeline
    q: list[float]

    def __init__(self, model_size: str = "base", **kwargs: Any) -> None:
        """
        Chronos を初期化し、事前学習済みモデルを読み込む。

        Parameters:
        ----------
        model_size : str, optional, default="base"
            モデルサイズ (例: 'tiny', 'mini', 'small', 'base', 'large')。
            "amazon/chronos-t5-{model_size}" として読み込まれる。
        **kwargs : Any
            Forecaster に渡す引数。
        """
        super().__init__(**kwargs)

        if type(self.prediction_length) == list:
            self.prediction_length = max(self.prediction_length)

        if type(self.context_length) == list:
            self.context_length = max(self.context_length)

        self.pred_len = self.prediction_length

        # Load pretrained model
        self.no_training = True

        self.pipeline = BaseChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{model_size}",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        self.q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Quantile levels

    def forecast(
        self, batch_data: "ProbTSBatchData", num_samples: int | None = None
    ) -> torch.Tensor:
        """
        Chronos パイプラインを用いて確率的予測サンプルを生成する。
        各変数を独立な単変量系列として内部バッチ単位で予測する。

        Parameters:
        ----------
        batch_data : ProbTSBatchData
            入力バッチデータ。
        num_samples : int | None, optional, default=None
            生成する予測サンプル数。

        Returns:
        ----------
        torch.Tensor
            予測サンプル。形状 [B, num_samples, pred_len, K]。
        """
        inputs = self.get_inputs(batch_data, "encode")
        inputs = inputs[:, -self.context_length :]

        B, _, K = inputs.shape
        inputs = rearrange(inputs, "b l k -> (b k) l")  # .cpu()
        context = [inputs[i] for i in range(B * K)]
        inner_batch_size = 12  # for 80G gpu
        forecast_samples = []

        # Process in batches of size `inner_batch_size`
        for i in range(0, len(context), inner_batch_size):
            batch_context = context[i : i + inner_batch_size]
            batch_forecast_samples = self.pipeline.predict(
                batch_context,
                prediction_length=self.pred_len,
                num_samples=num_samples,
                limit_prediction_length=False,
            )
            forecast_samples.append(batch_forecast_samples)

        forecast_samples = torch.cat(forecast_samples, dim=0)
        prob_forecast = rearrange(forecast_samples, "(b k) s l -> b s l k", b=B, k=K)

        return prob_forecast
