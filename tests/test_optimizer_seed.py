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


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------


def _cmaes_suggestions(seed: int) -> list[tuple[float, float]]:
    optimizer = OptunaOptimizer(sampler="cmaes", random_seed=seed)
    optimizer.initialize(num_goals=1)
    study = optimizer._get_study()
    suggestions = []
    for _ in range(12):  # past CMA-ES's first generation
        trial = study.ask()
        x, y = trial.suggest_float("x", -5.0, 5.0), trial.suggest_float("y", -5.0, 5.0, step=0.5)
        study.tell(trial, (x - 1) ** 2 + (y + 2) ** 2)
        suggestions.append((x, y))
    return suggestions


def test_cmaes_is_selected_by_name_with_the_margin_variant():
    import optuna

    optimizer = OptunaOptimizer(sampler="CmaEsSampler")
    optimizer.initialize(num_goals=1)
    sampler = optimizer._get_study().sampler

    assert isinstance(sampler, optuna.samplers.CmaEsSampler)
    assert sampler._with_margin


def test_a_seeded_cmaes_run_repeats():
    assert _cmaes_suggestions(7) == _cmaes_suggestions(7)
    assert _cmaes_suggestions(7) != _cmaes_suggestions(8)


def test_cmaes_finds_the_minimum_of_a_bowl_on_a_grid():
    import optuna

    optimizer = OptunaOptimizer(sampler="cmaes", random_seed=3)
    optimizer.initialize(num_goals=1)
    study = optimizer._get_study()
    study.optimize(
        lambda trial: (trial.suggest_float("x", -5.0, 5.0) - 1) ** 2
        + (trial.suggest_float("y", -5.0, 5.0, step=0.5) + 2) ** 2,
        n_trials=150,
    )

    assert study.best_value < 0.05
    assert study.best_params["y"] == -2.0
    assert isinstance(study.sampler, optuna.samplers.CmaEsSampler)


def test_cmaes_rejects_several_objectives():
    with pytest.raises(ConfigurationError, match="single objective"):
        OptunaOptimizer(sampler="cmaes", multi_objective=True).initialize(num_goals=2)
