"""Tests for the ``random_seed`` setting of the optimizers."""

from __future__ import annotations

import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.optimizers import GradientDescentOptimizer, OptunaOptimizer


def _first_suggestions(**settings) -> list[float]:
    optimizer = OptunaOptimizer(**settings)
    optimizer.initialize(num_goals=1)
    study = optimizer._get_study()
    return [study.ask().suggest_float("x", 0.0, 1.0) for _ in range(3)]


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_a_seeded_optuna_run_repeats(sampler):
    assert _first_suggestions(sampler=sampler, random_seed=7) == _first_suggestions(sampler=sampler, random_seed=7)
    assert _first_suggestions(sampler=sampler, random_seed=7) != _first_suggestions(sampler=sampler, random_seed=8)


def test_minus_one_means_no_seed():
    assert OptunaOptimizer(random_seed=-1).random_seed is None
    _first_suggestions(random_seed=-1)
    GradientDescentOptimizer(random_seed=-1)  # the setting documents -1 as no seed


def test_a_negative_seed_is_rejected():
    with pytest.raises(ConfigurationError, match="random_seed"):
        OptunaOptimizer(random_seed=-2)
