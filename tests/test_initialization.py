"""Tests for the starter configurations written by ``cobra init``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cobra.configuration.config_runner import build_configured_run
from cobra.configuration.configuration import ConfigurationError, RunConfiguration
from cobra.configuration.initialization import initial_configuration
from tests.conftest import MINIMAL_S2P, make_config_data, netlist_path


def test_ac_netlist_fills_sweep_and_lists_unmapped_components():
    initial = initial_configuration(netlist_path("minimal_ac"))

    config = initial.configuration
    assert config.simulation_parameters == {
        ".AC": {"points": "101", "start_freq": "1G", "stop_freq": "10G"}
    }
    assert config.component_models == {}
    assert config.design_goals == []
    assert config.optimization_parameters == []
    assert config.simulator.settings["xyce_command"] == "Xyce"
    assert initial.unmapped_components == ["X1"]
    assert "S21_dB" in initial.goal_parameters


def test_multi_tone_hb_frequencies_stay_in_one_parameter():
    initial = initial_configuration(netlist_path("hb_two_tone"))

    assert initial.configuration.simulation_parameters == {".HB": {"frequencies": "95E9 10E9"}}
    assert "Power_dBm[OUT]" in initial.goal_parameters


def test_given_models_are_resolved_and_checked(config_dir: Path):
    initial = initial_configuration(config_dir / "circuit.cir", {"X1": str(config_dir / "model.s2p")})

    assert initial.configuration.component_models == {"X1": str((config_dir / "model.s2p").resolve())}
    assert initial.unmapped_components == []

    with pytest.raises(ConfigurationError, match=r"Unknown component.*Y1.*components in circuit\.cir: X1"):
        initial_configuration(config_dir / "circuit.cir", {"Y1": "model.s2p"})


def test_existing_touchstone_file_of_the_netlist_is_used(tmp_path: Path):
    shutil.copy(netlist_path("tstonefile_qucs"), tmp_path / "trafo.cir")
    initial = initial_configuration(tmp_path / "trafo.cir")
    assert initial.unmapped_components == ["XTrafo1"]

    (tmp_path / "trafo.s6p").write_text(MINIMAL_S2P, encoding="utf-8")
    initial = initial_configuration(tmp_path / "trafo.cir")
    assert initial.configuration.component_models == {"XTrafo1": str(tmp_path / "trafo.s6p")}


def test_starter_configuration_round_trips_but_does_not_run(config_dir: Path):
    initial = initial_configuration(config_dir / "circuit.cir", {"X1": str(config_dir / "model.s2p")})
    reloaded = RunConfiguration.load(initial.configuration.save(config_dir / "starter.json"))

    assert reloaded.to_dict(config_dir) == initial.configuration.to_dict(config_dir)
    with pytest.raises(ConfigurationError, match="No design goals"):
        build_configured_run(reloaded)


def test_run_rejects_a_configuration_without_optimization_parameters(config_dir: Path):
    config = RunConfiguration.from_dict(make_config_data(optimization_parameters=[]), config_dir)

    with pytest.raises(ConfigurationError, match="No optimization parameters"):
        build_configured_run(config)
