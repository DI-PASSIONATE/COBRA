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
