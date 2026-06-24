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

        # Determine which simulation types are needed from the design goals
        design_goal_checker = context.get("design_goal_checker")
        required_types: Set[SimulationType] = set()
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
                netlist_path, sim_type, sim_params_by_type.get(sim_type, {}), results_dir
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

        # Merge GUI params over built-in defaults
        defaults = sim_type.positional_param_defaults()
        merged = {**defaults, **sim_params}

        # Directive not in netlist yet — inject it.
        # Build the positional token string in the correct order.
        tokens = [sim_type.value]
        for param_name in sim_type.positional_param_names():
            tokens.append(merged.get(param_name, ""))
        new_line = " ".join(t for t in tokens if t) + "\n"
        # Insert before the first .end / .END line, or append
        lines = parser._lines  # access internal line list
        end_idx = next(
            (i for i, l in enumerate(lines) if l.strip().upper() in {".END", ".ENDS"}),
            len(lines),
        )
        lines.insert(end_idx, new_line)

        parser.save(dest)
        return dest
