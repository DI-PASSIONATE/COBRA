"""Tests for analysis variants (a second analysis of a type) and expression goals."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cobra.configuration.config_runner import build_analysis_variants, build_design_goals
from cobra.configuration.configuration import (
    AnalysisVariantConfig,
    ConfigurationError,
    DesignGoalConfig,
    RunConfiguration,
)
from cobra.optimization_context import sanitize_for_json
from cobra.optimizers.design_goal import (
    FAILED_SIMULATION_PENALTY,
    DesignGoal,
    DesignGoalChecker,
    DesignParameter,
    ExpressionParameter,
)
from cobra.optimizers.design_goal_collection import (
    calculate_array_penalty,
    make_expression_parameter,
    make_gain_db,
    make_power_dbm,
    with_variant,
)
from cobra.optimizers.goal_expression import GoalExpression
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import AnalysisVariant, SimulationType
from cobra.spice_sim.vacask_simulator import VacaskSimulator
from cobra.stages.circuit_sim_stage import CircuitSimulationStage
from tests.conftest import make_context

if TYPE_CHECKING:
    from cobra.spice_sim.netlist_parsers.netlist import Netlist

# A 0.2 V sine source into a matched load, with an HB and a transient.
_PAD = """Matched source into a matched load
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
vp1 (a 0) vsource dc=0 mag=1 type="sine" ampl=0.2 freq=1G
rp1 (a out) r r=50
vout (out x) vsource dc=0
rload (x 0) r r=50
control
  analysis hb1 hb freq=[1G] nharm=3
  analysis tr1 tran step=10p stop=20n start=10n maxstep=10p
endc
"""

LOW_DRIVE = AnalysisVariant.create("HB_low", SimulationType.HB, {"vp1:ampl": 0.02})


def _netlist() -> Netlist:
    return VacaskSimulator.netlist_parser.parse(_PAD)


def _parameter(name: str, sim_type: SimulationType = SimulationType.HB) -> DesignParameter:
    """A parameter whose value is the number in its result's first output file name."""
    return DesignParameter(
        name=name,
        simulation_type=sim_type,
        formula=lambda result, _freq=None: np.array([float(result.output_files[0])]),
        loss=calculate_array_penalty,
    )


def _result(value: float) -> SimulationResult:
    return SimulationResult(output_files=[str(value)])


# ---------------------------------------------------------------------------
# AnalysisVariant and with_variant
# ---------------------------------------------------------------------------


def test_variant_sets_its_values_on_a_copy_of_the_netlist():
    netlist = _netlist()

    changed = LOW_DRIVE.apply(netlist)

    assert "ampl=0.02 " in changed.to_string()
    assert "ampl=0.2 " in netlist.to_string()
    assert LOW_DRIVE.netlist_value("VP1:AMPL") == "0.02"
    assert LOW_DRIVE.netlist_value("vp1:mag") is None


def test_with_variant_replaces_the_analysis_prefix():
    tran_low = AnalysisVariant.create("TRAN_low", SimulationType.TRAN, {"vp1:ampl": 0.02})
    gain = make_gain_db("vp1", 0.2, 50.0, "out", SimulationType.TRAN)

    moved = with_variant(gain, tran_low)

    assert gain.name == "TRAN:Gain_dB[vp1@out]"
    assert moved.name == "TRAN_low:Gain_dB[vp1@out]"
    assert moved.analysis == tran_low
    assert with_variant(make_power_dbm("out"), LOW_DRIVE).name == "HB_low:Power_dBm[out]"
    with pytest.raises(ValueError, match=r"needs a \.TRAN analysis"):
        with_variant(gain, LOW_DRIVE)
    with pytest.raises(ValueError, match="already read from the variant"):
        with_variant(moved, tran_low)


def test_context_export_names_a_variant_by_its_name():
    exported = sanitize_for_json({SimulationType.HB: 1, LOW_DRIVE: 2})

    assert exported == {".HB": 1, "HB_low": 2}


# ---------------------------------------------------------------------------
# DesignGoalChecker
# ---------------------------------------------------------------------------


def test_a_goal_on_a_variant_reads_the_variant_result():
    own = DesignGoal(_parameter("P"), max_value=10.0)
    low = DesignGoal(with_variant(_parameter("P"), LOW_DRIVE), max_value=10.0)
    checker = DesignGoalChecker([own, low])

    penalties = checker.loss({SimulationType.HB: _result(5.0), LOW_DRIVE: _result(20.0)})

    assert checker.required_analyses() == {SimulationType.HB, LOW_DRIVE}
    assert penalties[0] < 0 < penalties[1]
    assert low.current_value == pytest.approx([20.0])


def test_a_missing_variant_result_fails_only_its_goals():
    own = DesignGoal(_parameter("P"), max_value=10.0)
    low = DesignGoal(with_variant(_parameter("P"), LOW_DRIVE), max_value=10.0)

    penalties = DesignGoalChecker([own, low]).loss({SimulationType.HB: _result(5.0)})

    assert penalties[0] < 0
    assert penalties[1] == FAILED_SIMULATION_PENALTY


# ---------------------------------------------------------------------------
# GoalExpression
# ---------------------------------------------------------------------------


def test_expression_combines_values_elementwise():
    expression = GoalExpression("10 * log10(max(a, b) / 2) - -c ** 2")
    values = {"a": np.array([20.0, 2.0]), "b": np.array([4.0, 200.0]), "c": np.array(3.0)}

    assert expression.variables == ("a", "b", "c")
    assert expression.evaluate(values) == pytest.approx([19.0, 29.0])
    assert GoalExpression("min(a, b, 1)").evaluate(values) == pytest.approx([1.0, 1.0])
    assert GoalExpression("abs(-a) + sqrt(4) + exp(0) + ln(1)").evaluate({"a": np.array(1.0)}) == 4.0


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("a -", "not valid"),
        ("a < b", "is not allowed"),
        ("a.real", "is not allowed"),
        ("__import__('os')", "Unknown function"),
        ("'a' + b", "is not allowed"),
        ("True + a", "is not allowed"),
        ("min(a)", "takes at least 2"),
        ("sqrt(a, b)", "takes 1"),
        ("abs", "is a function"),
        ("max(*a, b)", "is not allowed"),
        ("ln(x=a)", "is not allowed"),
    ],
)
def test_expression_outside_the_grammar_is_rejected(text, match):
    with pytest.raises(ValueError, match=match):
        GoalExpression(text)


# ---------------------------------------------------------------------------
# ExpressionParameter
# ---------------------------------------------------------------------------


def _compression(max_value: float = 1.0) -> DesignGoal:
    parameter = make_expression_parameter(
        "Compression", "a - b", {"a": _parameter("G"), "b": with_variant(_parameter("G"), LOW_DRIVE)}
    )
    return DesignGoal(parameter, frequency_range="1GHz", max_value=max_value)


def test_expression_goal_combines_its_terms_across_analyses():
    goal = _compression()
    checker = DesignGoalChecker([goal])

    penalty = goal.evaluate({SimulationType.HB: _result(3.0), LOW_DRIVE: _result(2.2)})

    assert goal.parameter_name == "Compression"
    assert goal.current_value == pytest.approx([0.8])
    assert penalty < 0
    assert goal.required_analyses == (SimulationType.HB, LOW_DRIVE)
    assert checker.required_analyses() == {SimulationType.HB, LOW_DRIVE}
    assert checker.goals_reading(LOW_DRIVE) == [goal]


def test_expressions_chain():
    compression = _compression().parameter
    relative = make_expression_parameter("Relative", "100 * c / g", {"c": compression, "g": _parameter("G")})

    values = relative.values({SimulationType.HB: _result(4.0), LOW_DRIVE: _result(3.0)})

    assert values == pytest.approx([25.0])
    assert relative.required_analyses == (SimulationType.HB, LOW_DRIVE)
    with pytest.raises(ValueError, match="depends on itself"):
        make_expression_parameter("Compression", "2 * r", {"r": relative})


def test_expression_goal_fails_without_a_result_or_a_finite_value():
    goal = _compression()

    assert goal.evaluate({SimulationType.HB: _result(3.0)}) == FAILED_SIMULATION_PENALTY
    assert goal.current_value is None
    root = DesignGoal(make_expression_parameter("Root", "sqrt(a)", {"a": _parameter("G")}), max_value=1.0)
    assert root.evaluate({SimulationType.HB: _result(-1.0)}) == FAILED_SIMULATION_PENALTY


def test_expression_rejects_terms_of_different_length():
    band = DesignParameter(
        "Band", SimulationType.HB, lambda _result, _freq=None: np.zeros(3), calculate_array_penalty
    )
    parameter = make_expression_parameter("Odd", "a - b", {"a": _parameter("G"), "b": band})
    two = DesignParameter("Two", SimulationType.HB, lambda _r, _f=None: np.zeros(2), calculate_array_penalty)

    # A single value broadcasts; two different lengths do not.
    assert parameter.values({SimulationType.HB: _result(3.0)}) == pytest.approx([3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="single frequency"):
        make_expression_parameter("Odd2", "a - b", {"a": two, "b": band}).values({SimulationType.HB: _result(0)})


@pytest.mark.parametrize(
    ("expression", "terms", "match"),
    [
        ("a - b", {"a": "G"}, "uses b, which are not among its terms"),
        ("a", {"a": "G", "b": "G"}, "does not use its terms b"),
        ("1", {}, "at least one term"),
    ],
)
def test_expression_terms_must_match_its_variables(expression, terms, match):
    with pytest.raises(ValueError, match=match):
        make_expression_parameter("E", expression, {key: _parameter(name) for key, name in terms.items()})


# ---------------------------------------------------------------------------
# CircuitSimulationStage
# ---------------------------------------------------------------------------


class _RecordingSimulator(VacaskSimulator):
    """Prepares netlists as VACASK does and records what it would run."""

    def __init__(self):
        super().__init__()
        self.runs: dict[str, str] = {}

    def run_simulation(self, netlist_path: str, netlist: Netlist) -> SimulationResult | None:
        self.runs[Path(netlist_path).name] = netlist.to_string()
        return SimulationResult(output_files=[netlist_path])


def test_stage_runs_a_variant_on_its_own_netlist(tmp_path):
    deck = tmp_path / "deck.sim"
    deck.write_text(_PAD, encoding="utf-8")
    goals = [DesignGoal(make_power_dbm("out"), max_value=0.0)]
    goals.append(DesignGoal(with_variant(make_power_dbm("out"), LOW_DRIVE), max_value=0.0))
    context = make_context(
        netlist=str(deck),
        results_dir=str(tmp_path),
        native_sim_type=SimulationType.HB,
        design_goal_checker=DesignGoalChecker(goals),
    )
    simulator = _RecordingSimulator()

    context = CircuitSimulationStage(simulator).run(context)

    assert set(context.simulation_results) == {SimulationType.HB, LOW_DRIVE}
    assert set(simulator.runs) == {"deck_hb.sim", "deck_hb_low.sim"}
    assert "ampl=0.2 " in simulator.runs["deck_hb.sim"]
    assert "ampl=0.02 " in simulator.runs["deck_hb_low.sim"]
    # The variant runs its base analysis alone, like the netlist's own.
    assert "//" in simulator.runs["deck_hb_low.sim"].split("analysis tr1")[0].splitlines()[-1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"name": "1st", "analysis": "HB", "netlist_values": {"vp1:ampl": 1}}, "identifier"),
        ({"name": "Tran", "analysis": "HB", "netlist_values": {"vp1:ampl": 1}}, "is an analysis type"),
        ({"name": "HB_low", "analysis": "PSS", "netlist_values": {"vp1:ampl": 1}}, "analysis must be one of"),
        ({"name": "HB_low", "analysis": "HB", "netlist_values": {}}, "sets no netlist_values"),
        ({"name": "HB_low", "analysis": "HB", "netlist_values": {"vp1:ampl": " "}}, "is empty"),
        ({"name": "HB_low", "analysis": "HB", "netlist_values": {"vp1:ampl": True}}, "finite number"),
    ],
)
def test_invalid_analysis_variant_is_rejected(data, match):
    with pytest.raises(ConfigurationError, match=match):
        AnalysisVariantConfig.from_dict(data).validate()


_VARIANT = {"name": "HB_low", "analysis": "HB", "netlist_values": {"vp1:ampl": 0.02}}
_POWER = {"parameter": "Power_dBm[out]", "kind": "power_dbm", "node": "out", "max_value": 0.0}
_LOW_POWER = {**_POWER, "parameter": "HB_low:Power_dBm[out]", "analysis": "HB_low"}
_GAIN = {
    "parameter": "Gain_dB[vp1@out]", "kind": "gain_db", "node": "out", "port": "vp1",
    "source_amplitude": 0.2, "impedance": 50.0, "frequency_range": "1GHz", "min_value": -10.0,
}
_LOW_GAIN = {**_GAIN, "parameter": "HB_low:Gain_dB[vp1@out]", "analysis": "HB_low", "source_amplitude": 0.02}
_COMPRESSION = {
    "parameter": "Compression_dB",
    "kind": "expression",
    "expression": "g - g_low",
    "terms": {"g": "Gain_dB[vp1@out]", "g_low": "HB_low:Gain_dB[vp1@out]"},
    "frequency_range": "1GHz",
    "max_value": 1.0,
}
# Chained: the compression relative to the gain, in percent.
_RELATIVE = {
    "parameter": "Compression_pct",
    "kind": "expression",
    "expression": "100 * c / g",
    "terms": {"c": "Compression_dB", "g": "Gain_dB[vp1@out]"},
    "frequency_range": "1GHz",
    "max_value": 10.0,
}


def _config(config_dir: Path, config_data: dict, goals: list[dict], variants=(_VARIANT,)) -> RunConfiguration:
    return RunConfiguration.from_dict(
        {**config_data, "analysis_variants": list(variants), "design_goals": goals}, config_dir
    )


def _build(goals: list[dict], variants=(_VARIANT,)) -> list[DesignGoal]:
    netlist = _netlist()
    built = build_analysis_variants([AnalysisVariantConfig.from_dict(v) for v in variants], netlist)
    return build_design_goals([DesignGoalConfig.from_dict(g) for g in goals], netlist, None, built)


def test_variants_and_expression_goals_round_trip(config_dir, config_data):
    config = _config(config_dir, config_data, [_GAIN, _LOW_GAIN, _COMPRESSION, _RELATIVE])

    reloaded = RunConfiguration.load(config.save(config_dir / "run.json"))

    assert reloaded.analysis_variants == config.analysis_variants
    assert reloaded.design_goals[2].terms == _COMPRESSION["terms"]
    assert reloaded.design_goals[2].expression == "g - g_low"


@pytest.mark.parametrize(
    ("goals", "variants", "match"),
    [
        ([{**_POWER, "analysis": "HB_high"}], [_VARIANT], "or an analysis variant \\(HB_low\\)"),
        ([_LOW_POWER], [_VARIANT, _VARIANT], "names must be unique"),
        ([_GAIN, _COMPRESSION], [_VARIANT], "no other design goal reads"),
        ([_GAIN, {**_COMPRESSION, "terms": ["Gain_dB[vp1@out]"]}], [_VARIANT], "needs terms"),
        ([_GAIN, _LOW_GAIN, {**_COMPRESSION, "expression": "g -"}], [_VARIANT], "is not valid"),
        ([_GAIN, _LOW_GAIN, {**_COMPRESSION, "expression": "g"}], [_VARIANT], "unused g_low"),
        ([_GAIN, _LOW_GAIN, {**_COMPRESSION, "expression": "g - h"}], [_VARIANT], "missing h"),
        ([_GAIN, {**_POWER, "expression": "a"}], [_VARIANT], "Only an expression goal"),
        ([_GAIN, _LOW_GAIN, _COMPRESSION, _COMPRESSION], [_VARIANT], "shares its name"),
        (
            [_GAIN, _LOW_GAIN, {**_COMPRESSION, "terms": {"g": "Compression_pct", "g_low": "HB_low:Gain_dB[vp1@out]"}}, _RELATIVE],
            [_VARIANT],
            "depends on itself",
        ),
        (
            [{**_POWER, "analysis": "AC_2"}],
            [{"name": "AC_2", "analysis": "AC", "netlist_values": {"R1": 1}}],
            "runs AC, not HB or TRAN",
        ),
    ],
)
def test_invalid_variant_goal_configuration_is_rejected(config_dir, config_data, goals, variants, match):
    with pytest.raises(ConfigurationError, match=match):
        _config(config_dir, config_data, goals, variants)


def test_variant_goals_and_chained_expressions_are_built():
    # The chained expression comes first; it is built after the one it uses.
    relative, gain, low_gain, compression = _build([_RELATIVE, _GAIN, _LOW_GAIN, _COMPRESSION])

    assert low_gain.parameter.analysis == LOW_DRIVE
    assert low_gain.parameter_name == "HB_low:Gain_dB[vp1@out]"
    assert isinstance(compression.parameter, ExpressionParameter)
    assert compression.parameter.terms == {"g": gain.parameter, "g_low": low_gain.parameter}
    assert isinstance(relative.parameter, ExpressionParameter)
    assert relative.parameter.terms["c"] is compression.parameter
    assert relative.required_analyses == (SimulationType.HB, LOW_DRIVE)


def test_a_gain_goal_on_a_variant_takes_the_variant_drive():
    with pytest.raises(ConfigurationError, match=r"drives 'vp1' with 0\.02"):
        _build([{**_LOW_GAIN, "source_amplitude": 0.2}])


def test_a_variant_value_must_name_a_netlist_element():
    with pytest.raises(ConfigurationError, match="no element or instance 'vp9'"):
        _build([], [{**_VARIANT, "netlist_values": {"vp9:ampl": 0.02}}])


def test_a_catalogue_goal_on_a_variant_needs_its_analysis_type():
    s11 = {"parameter": "HB_low:S11_dB", "max_value": -10.0, "analysis": "HB_low"}

    with pytest.raises(ConfigurationError, match="needs a AC analysis"):
        _build([s11])


def test_an_expression_across_analyses_needs_a_single_frequency():
    band = {**_COMPRESSION, "frequency_range": "1-2GHz"}

    with pytest.raises(ConfigurationError, match="single frequency"):
        _build([_GAIN, _LOW_GAIN, band])


def test_expressions_in_a_cycle_are_not_built():
    first = {**_COMPRESSION, "parameter": "A", "expression": "b", "terms": {"b": "B"}}
    second = {**_COMPRESSION, "parameter": "B", "expression": "a", "terms": {"a": "A"}}

    with pytest.raises(ConfigurationError, match="cycle"):
        _build([first, second])
