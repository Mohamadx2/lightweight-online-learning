"""
Online / incremental learning logic.

This module will contain the core update mechanisms:
- Adding new labeled examples to a retrieval store
- Optionally updating a lightweight classification/ranking head
- Tracking performance over time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    text: str
    label: str | int
    metadata: dict[str, Any] = field(default_factory=dict)


class OnlineLearner:
    """
    Minimal interface for an online learner.

    Concrete implementations can use pure retrieval, a linear head on
    frozen embeddings, or low-rank adapters.
    """

    def __init__(self):
        self.examples: list[Example] = []

    def add_examples(self, examples: list[Example]) -> None:
        """Incorporate new labeled examples."""
        self.examples.extend(examples)
        # TODO: update vector store / adapter weights

    def predict(self, text: str) -> Any:
        """Return a prediction for the given text."""
        raise NotImplementedError

    def explain(self, text: str, k: int = 3) -> list[Example]:
        """Return the most influential past examples for a prediction."""
        raise NotImplementedError
