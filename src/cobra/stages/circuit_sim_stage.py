from typing import Dict, List, Set
import os
import shutil

from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.spice_sim.simulation_type import SimulationType
from cobra.stages.base_stage import COBRABaseStage
import skrf as rf


class CircuitSimulationStage(COBRABaseStage):
    """
    Circuit Simulation Stage — runs one Xyce simulation per required analysis
    type and stores all results in ``context["simulated_networks"]``.

    """

    def __init__(self, simulator: BaseSimulator = XyceSimulator("Xyce")):
        self.simulator = simulator

    def run(self, context: Dict) -> Dict:
        ntwks: List[rf.Network] = context["predicted_networks"]
        results_dir = context.get("results_dir", ".")

        # Preprocess surrogate models (e.g. vector fitting)
        for n in ntwks:
            out_name = os.path.join(results_dir, n.name if n.name else "cobra_output")
            self.simulator.preprocess_ntwk(n, name=out_name)

        # Determine which simulation types to run:
        # 1. Always run the netlist's native simulation type.
        # 2. Also run any type required by a design goal (e.g. AC for S-params
        #    when the native type is HB).
        native_sim_type: SimulationType = context.get("native_sim_type", SimulationType.UNKNOWN)
        required_types: Set[SimulationType] = set()
        if native_sim_type is not SimulationType.UNKNOWN:
            required_types.add(native_sim_type)

        design_goal_checker = context.get("design_goal_checker")
        if design_goal_checker:
            for goal in design_goal_checker.design_goals:
                st = goal.required_simulation_type
                if st is not SimulationType.UNKNOWN:
                    required_types.add(st)
        if not required_types:
            required_types = {SimulationType.AC}  # sensible default

        # Per-type simulation parameters from the GUI (e.g. sweep range edits)
        sim_params_by_type: Dict[SimulationType, Dict[str, str]] = context.get("sim_params_by_type", {})

        netlist_path: str = context["netlist"]
        simulated_networks: Dict[SimulationType, rf.Network] = {}

        for sim_type in required_types:
            prepared = self._prepare_netlist_for_type(
                netlist_path, sim_type, sim_params_by_type.get(sim_type, {}), results_dir,
                simulator=self.simulator,
            )
            ntwk_result = self.simulator.run_simulation(prepared)
            if ntwk_result is not None:
                simulated_networks[sim_type] = ntwk_result

        context["simulated_networks"] = simulated_networks

        return context

    @staticmethod
    def _prepare_netlist_for_type(
        netlist_path: str,
        sim_type: SimulationType,
        sim_params: Dict[str, str],
        results_dir: str,
        simulator: "BaseSimulator",
    ) -> str:
        """
        Return the path to a netlist ready for *sim_type*.

        If the netlist already contains a directive matching *sim_type*, it is
        returned unchanged (the existing directive is assumed correct).

        Otherwise a copy is placed in *results_dir* with the appropriate
        directive injected using *sim_params* (falling back to the type's built-in
        defaults for any missing parameter).
        """
        # Import here to avoid a top-level circular dependency
        from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser

        parser = XyceNetlistParser().from_file(netlist_path)
        existing = [d for d in parser.simulation_directives
                    if SimulationType.from_directive(d.directive) is sim_type]
        if existing:
            return netlist_path

        base = os.path.basename(netlist_path)
        name, ext = os.path.splitext(base)
        dest = os.path.join(results_dir, f"{name}_{sim_type.name.lower()}{ext}")

        # Merge GUI params over built-in defaults from the simulator
        meta = simulator.get_simulation_metadata(sim_type)
        defaults = meta.positional_param_defaults
        merged = {**defaults, **sim_params}

        # Build the new directive line(s).
        tokens = [sim_type.value]
        for param_name in meta.positional_param_names:
            # A param value may contain multiple space-separated tokens
            # (e.g. HB frequencies = "95E9 10E9") — expand them individually.
            raw = merged.get(param_name, "")
            tokens.extend(t for t in raw.split() if t)
        new_directive = " ".join(t for t in tokens if t) + "\n"

        # For AC we also need .LIN so Xyce writes a Touchstone file.
        extra_lines = []
        if sim_type is SimulationType.AC:
            extra_lines.append(".LIN format=touchstone sparcalc=1\n")

        # Work on a copy of the raw lines.
        lines = parser._lines[:]

        # Remove all existing top-level simulation directives and their
        # companion lines (.PRINT, .options hbint, etc.) that belong to
        # a different analysis — they must not appear in the copy netlist.
        # Track subckt nesting so we only remove top-level directives.
        _SIM_PREFIXES = {".hb", ".tran", ".dc", ".ac", ".lin",
                         ".print", ".options", ".measure", ".four"}
        pruned: list[str] = []
        depth = 0
        for raw in lines:
            stripped = raw.strip().lower()
            if stripped.startswith(".subckt"):
                depth += 1
                pruned.append(raw)
                continue
            if stripped.startswith(".ends"):
                if depth > 0:
                    depth -= 1
                pruned.append(raw)
                continue
            # At top level: drop lines that start with a simulation directive
            if depth == 0 and any(stripped.startswith(p) for p in _SIM_PREFIXES):
                continue
            pruned.append(raw)

        # Insert new directive(s) before the top-level .END line.
        end_idx = next(
            (i for i, l in enumerate(pruned)
             if l.strip().upper() == ".END"),
            len(pruned),
        )
        for extra in reversed(extra_lines):
            pruned.insert(end_idx, extra)
        pruned.insert(end_idx, new_directive)

        parser._lines = pruned
        parser.save(dest)
        return dest
