"""Loading a run configuration into the main window, headless (Qt's offscreen platform)."""

from __future__ import annotations

import pytest

from cobra.configuration.configuration import ConfigurationError, RunConfiguration
from tests.conftest import MINIMAL_S2P, make_config_data

# X1 is mapped to a model, X2 is not: loading fails after the netlist is parsed.
_DECK = """Two unmodelled subcircuits
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
X1 (in mid) surrogate_a
X2 (mid out) surrogate_b
vp1 (a1 0) vsource dc=0 mag=1
rp1 (a1 in) r r=50
vp2 (a2 0) vsource dc=0
rp2 (a2 out) r r=50
control
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1G to=3G mode="lin" points=3
endc
"""


def test_a_failed_load_still_shows_the_configured_simulator(tmp_path, window):
    (tmp_path / "deck.sim").write_text(_DECK, encoding="utf-8")
    (tmp_path / "model.s2p").write_text(MINIMAL_S2P, encoding="utf-8")
    data = make_config_data(
        netlist="deck.sim",
        simulation_parameters={},
        simulator={"name": "VacaskSimulator", "settings": {}},
    )
    config = RunConfiguration.from_dict(data, tmp_path)

    with pytest.raises(ConfigurationError, match="Missing component models: X2"):
        window.apply_run_configuration(config)
    assert window.simulator_combo.currentText() == "VacaskSimulator"
    assert "vector_fit_max_order" in window.simulator_widgets


# A driven pad around one surrogate, with an S-parameter and an HB analysis.
_PAD = """Driven pad
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
vp1 (a 0) vsource dc=0 mag=1 type="sine" ampl=0.2 freq=1G
rp1 (a in) r r=50
X1 (in out) surrogate_a
vout (out x) vsource dc=0
rload (x 0) r r=50
control
  analysis sp1 acsp ports=["vp1", "rp1"] from=1G to=2G mode="lin" points=2
  analysis hb1 hb freq=[1G] nharm=3
endc
"""

_GAIN = {
    "parameter": "Gain_dB[vp1@out]", "kind": "gain_db", "node": "out", "port": "vp1",
    "source_amplitude": 0.2, "impedance": 50.0, "frequency_range": "1GHz", "min_value": -10.0,
}
_LOW_GAIN = {**_GAIN, "parameter": "HB_low:Gain_dB[vp1@out]", "analysis": "HB_low", "source_amplitude": 0.02}
_COMPRESSION = {
    "parameter": "Compression_dB", "kind": "expression", "expression": "g - g_low",
    "terms": {"g": "Gain_dB[vp1@out]", "g_low": "HB_low:Gain_dB[vp1@out]"},
    "frequency_range": "1GHz", "max_value": 1.0,
}


def _variant_config(tmp_path) -> RunConfiguration:
    (tmp_path / "pad.sim").write_text(_PAD, encoding="utf-8")
    (tmp_path / "model.s2p").write_text(MINIMAL_S2P, encoding="utf-8")
    data = make_config_data(
        netlist="pad.sim",
        simulation_parameters={},
        simulator={"name": "VacaskSimulator", "settings": {}},
        optimization_parameters=[
            {"name": "rload:r", "type": "netlist_variable", "min_value": 40.0, "max_value": 60.0, "step": 1.0},
        ],
        analysis_variants=[{"name": "HB_low", "analysis": "HB", "netlist_values": {"vp1:ampl": 0.02}}],
        design_goals=[_GAIN, _LOW_GAIN, _COMPRESSION],
    )
    return RunConfiguration.from_dict(data, tmp_path)


def _goal_entries(config: RunConfiguration) -> list[tuple]:
    return [
        (goal.parameter, goal.kind, goal.analysis, goal.source_amplitude, goal.expression, goal.terms)
        for goal in config.design_goals
    ]


def test_variants_and_expression_goals_survive_the_window(tmp_path, window):
    config = _variant_config(tmp_path)

    window.apply_run_configuration(config)
    saved = window.create_run_configuration()

    assert saved.analysis_variants == config.analysis_variants
    assert _goal_entries(saved) == _goal_entries(config)
    assert window.variant_list.count() == 1
    offered = {parameter.name for parameter in window._available_parameters}
    assert {"HB_low:Gain_dB[vp1@out]", "HB_low:Power_dBm[out]"} <= offered


def test_editing_a_variant_moves_its_goals_to_the_new_drive(tmp_path, window):
    window.apply_run_configuration(_variant_config(tmp_path))
    [variant] = window.analysis_variants

    window._set_analysis_variants([type(variant)("HB_low", "HB", {"vp1:ampl": 0.05})])

    goals = window.create_run_configuration().design_goals
    assert goals[1].source_amplitude == 0.05
    assert _expression(window.goals[2]).terms["g_low"] is window.goals[1].parameter


def test_goals_and_variants_in_use_are_kept(tmp_path, window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    told: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: told.append(args[2]))
    window.apply_run_configuration(_variant_config(tmp_path))

    window.delete_goal(window.goal_list.item(0))
    window.delete_analysis_variant(0)

    assert len(window.goals) == 3
    assert len(window.analysis_variants) == 1
    assert "uses Gain_dB[vp1@out], which no other design goal reads" in told[0]
    assert "read HB_low" in told[1]


def test_editing_a_term_updates_the_expressions_using_it(tmp_path, window):
    from cobra.gui.dialogs import DesignGoalDialog

    window.apply_run_configuration(_variant_config(tmp_path))
    # A chained expression on the compression.
    window._apply_goals([*window.goals, _dialog_goal(window, "Twice", "2 * c", min_value=None)])
    compression = window.goals[2]
    dialog = DesignGoalDialog(
        window,
        goal=compression,
        available_parameters=window._available_parameters,
        term_parameters=window._term_parameters(exclude=compression),
        taken_names=window._taken_names(exclude=compression),
    )
    # The goal's own terms keep their variables; an expression using it is not offered.
    terms = [dialog.terms_list.item(i).text() for i in range(dialog.terms_list.count())]
    assert terms[:2] == ["g = Gain_dB[vp1@out]", "g_low = HB_low:Gain_dB[vp1@out]"]
    assert not any("Twice" in term for term in terms)
    dialog.expression_edit.setText("g_low - g")

    assert window._apply_goals([*window.goals[:2], dialog.get_data(), window.goals[3]])
    assert _expression(window.goals[3]).terms["c"] is window.goals[2].parameter
    assert str(_expression(window.goals[2]).expression) == "GoalExpression('g_low - g')"


def _expression(goal):
    from cobra.optimizers.design_goal import ExpressionParameter

    assert isinstance(goal.parameter, ExpressionParameter)
    return goal.parameter


def _problem(dialog) -> str:
    problem = dialog._validate()
    assert problem is not None
    return problem[0]


def _dialog_goal(window, name: str, expression: str, min_value: float | None = 0.0):
    from cobra.gui.dialogs import DesignGoalDialog

    dialog = DesignGoalDialog(
        window,
        available_parameters=window._available_parameters,
        term_parameters=window._term_parameters(),
        taken_names=window._taken_names(),
    )
    tabs = [dialog.sim_type_tabs.tabText(i) for i in range(dialog.sim_type_tabs.count())]
    dialog.sim_type_tabs.setCurrentIndex(tabs.index("Expression"))
    dialog.name_edit.setText(name)
    dialog.expression_edit.setText(expression)
    dialog.max_edit.setText("1.0")
    if min_value is not None:
        dialog.min_edit.setText(str(min_value))
    dialog.freq_mode_combo.setCurrentText("Single frequency")
    dialog.freq_min_edit.setText("1")
    return dialog.get_data()


def test_goal_dialog_offers_variant_tabs_and_an_expression(tmp_path, window):
    from cobra.gui.dialogs import DesignGoalDialog

    window.apply_run_configuration(_variant_config(tmp_path))
    dialog = DesignGoalDialog(
        window,
        available_parameters=window._available_parameters,
        term_parameters=window._term_parameters(),
        taken_names=window._taken_names(),
    )
    tabs = [dialog.sim_type_tabs.tabText(i) for i in range(dialog.sim_type_tabs.count())]
    assert "HB_low" in tabs
    dialog.sim_type_tabs.setCurrentIndex(tabs.index("Expression"))
    terms = [dialog.terms_list.item(i).text() for i in range(dialog.terms_list.count())]
    assert terms == ["a = Gain_dB[vp1@out]", "b = HB_low:Gain_dB[vp1@out]", "c = Compression_dB"]
    dialog.max_edit.setText("1.0")

    # A name another goal has, an unknown variable, and several analyses over a band are refused.
    dialog.name_edit.setText("Compression_dB")
    dialog.expression_edit.setText("a - b")
    assert "choose another name" in _problem(dialog)
    dialog.name_edit.setText("Drop")
    dialog.expression_edit.setText("a - z")
    assert "z is not a term" in _problem(dialog)
    dialog.terms_list.itemDoubleClicked.emit(dialog.terms_list.item(1))
    dialog.expression_edit.setText("a - b")
    dialog.freq_mode_combo.setCurrentText("Whole sweep / spectrum")
    assert "single frequency" in _problem(dialog)
    dialog.freq_mode_combo.setCurrentText("Single frequency")
    dialog.freq_min_edit.setText("1")
    goal = dialog.get_data()

    assert goal.parameter_name == "Drop"
    assert _expression(goal).terms == {"a": window.goals[0].parameter, "b": window.goals[1].parameter}


def test_retune_settings_survive_the_window(tmp_path, window):
    config = _variant_config(tmp_path)
    config.fine_tuning.retune_trials = 150
    config.fine_tuning.retune_range = 0.25
    config.fine_tuning.retune_min_steps = 3

    window.apply_run_configuration(config)
    saved = window.create_run_configuration().fine_tuning

    assert (saved.retune_trials, saved.retune_range, saved.retune_min_steps) == (150, 0.25, 3)
    window.retune_trials_spin.setValue(0)
    assert not window.retune_range_spin.isEnabled()


def test_rarely_changed_settings_are_behind_advanced(window):
    window.simulator_combo.setCurrentText("VacaskSimulator")

    advanced = window.simulator_section.advanced_form
    basic = window.simulator_section.form
    assert basic.getWidgetPosition(window.simulator_widgets["vacask_command"])[0] >= 0
    assert advanced.getWidgetPosition(window.simulator_widgets["vector_fit_max_order"])[0] >= 0
    assert window.optimizer_section.advanced_form.getWidgetPosition(window.optimizer_widgets["random_seed"])[0] >= 0
    assert window.finetune_section.advanced_form.getWidgetPosition(window.retune_range_spin)[0] >= 0

    # Switching back rebuilds both forms without leftovers.
    window.simulator_combo.setCurrentText("XyceSimulator")
    rows = basic.rowCount() + advanced.rowCount()
    window.simulator_combo.setCurrentText("VacaskSimulator")
    window.simulator_combo.setCurrentText("XyceSimulator")
    assert basic.rowCount() + advanced.rowCount() == rows


def test_collapsed_sections_summarise_their_settings(tmp_path, window):
    config = _variant_config(tmp_path)
    config.fine_tuning.enabled = True
    config.fine_tuning.retune_trials = 200
    config.parallel_trials = 4

    window.apply_run_configuration(config)
    for section in window._sections:
        section.set_expanded(False)

    assert window.circuit_section.summary_label.text() == "pad.sim · 1 component"
    assert window.analysis_section.summary_label.text() == ".AC, .HB · analysis point out"
    assert window.optimizer_section.summary_label.text() == (
        "OptunaOptimizer · up to 10 iterations · 4 parallel trials"
    )
    assert window.finetune_section.summary_label.text().endswith("3 iterations · retune 200 trials")
    assert not window.finetune_section.summary_label.isHidden()


def test_switching_fine_tuning_opens_its_section(window):
    window.finetune_cb.setChecked(True)
    assert window.finetune_section.expanded
    assert window.finetune_section.body.isEnabled()

    window.finetune_cb.setChecked(False)
    assert not window.finetune_section.expanded
    assert not window.finetune_section.body.isEnabled()
    assert window.finetune_section.summary_label.text() == "Off"
