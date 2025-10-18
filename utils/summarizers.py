from abc import ABC, abstractmethod
from typing import Any

import matplotlib.figure as fg
import matplotlib.pyplot as plt
from summarizer import Summarizer
from transformers import pipeline

from utils.cache import RedisCache, wrap_cache


class BaseEnum:
    @classmethod
    def values(cls) -> list[Any]:
        return [v for k, v in vars(cls).items() if not k.startswith("__")]


class ExtractiveAllowedModels(BaseEnum):
    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    XLNET_BASE = "xlnet-base-cased"
    XLM_MLM = "xlm-mlm-enfr-1024"
    DISTILBERT_BASE = "distilbert-base-uncased"
    ALBERT_BASE = "albert-base-v1"
    ALBERT_LARGE = "albert-large-v1"


class AbstractiveAllowedModels(BaseEnum):
    BART = "facebook/bart-large-cnn"
    PEGASUS = "google/pegasus-xsum"
    T5_BASE = "t5-base"


class BaseSummarizerModel(ABC):
    allowed_models: type[BaseEnum] | None = None

    def __init__(self, model_name: str, text: str) -> None:
        if self.allowed_models is None:
            raise ValueError("Allowed models variable is not set")
        else:
            if model_name not in self.allowed_models.values():
                raise KeyError(f"Model {model_name} not allowed in {self.allowed_models.__class__.__name__}")

        self.model_name = model_name
        if text is not None and "." in text:
            self.text = text
        else:
            raise ValueError("Fill text correctly")

    def __str__(self) -> str:
        return f"{self.model_name}::{hash(self.text)}"

    @abstractmethod
    def generate_summarize(self, *args, **kwargs) -> str:
        pass


class ExtractiveSummarizerModel(BaseSummarizerModel):
    allowed_models = ExtractiveAllowedModels

    def __init__(self, model_name: str, text: str) -> None:
        super().__init__(model_name, text)
        self.model = Summarizer(model=model_name)

    @wrap_cache(cache=RedisCache)
    def generate_summarize(self, ratio: float = 0.3) -> str:
        return self.model(self.text, ratio=ratio)

    @wrap_cache(cache=RedisCache)
    def calculate_elbow(self, k_max=10) -> tuple[list[float], int]:
        values = self.model.calculate_elbow(self.text, k_max=k_max)
        optimal_value = self.model.calculate_optimal_k(self.text, k_max=k_max)

        return values, optimal_value

    def plot_elbow(self) -> fg.Figure:
        values, optimal_value = self.calculate_elbow()

        fig, ax = plt.subplots()
        ax.plot(range(1, len(values) + 1), values, marker="o", linestyle="-")
        ax.scatter(optimal_value + 1, values[optimal_value], color="red", zorder=3)
        ax.set_title("Elbow method for optimal cluster count")
        ax.set_xlabel("N clusters")
        ax.set_ylabel("Metric value")
        return fig


class AbstractiveSummarizerModel(BaseSummarizerModel):
    allowed_models = AbstractiveAllowedModels

    def __init__(self, model_name: str, text: str) -> None:
        super().__init__(model_name, text)
        self.model = pipeline("summarization", model=model_name, device="cpu")

    @wrap_cache(cache=RedisCache)
    def generate_summarize(self, max_length: int = 130, min_length: int = 50) -> str:
        summarize = self.model(self.text, max_length=max_length, min_length=min_length, do_sample=False)
        return summarize[0]["summary_text"]
