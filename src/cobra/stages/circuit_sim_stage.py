import logging
import os
from typing import TYPE_CHECKING

from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.base_stage import COBRABaseStage

if TYPE_CHECKING:
    import skrf as rf

    from cobra.optimization_context import OptimizationContext

logger = logging.getLogger(__name__)


class CircuitSimulationStage(COBRABaseStage):
    """
    Circuit Simulation Stage — runs one Xyce simulation per required analysis
    type and stores all results in ``context.simulation_results``.

    """

    def __init__(self, simulator: BaseSimulator | None = None):
        self.simulator = simulator if simulator is not None else XyceSimulator("Xyce")

    def run(self, context: "OptimizationContext") -> "OptimizationContext":
        ntwks: list[rf.Network] = context.predicted_networks
        results_dir = context.results_dir

        # Preprocess surrogate models (e.g. vector fitting)
        for n in ntwks:
            out_name = os.path.join(results_dir, n.name or "cobra_output")
            self.simulator.preprocess_ntwk(n, name=out_name)

        # Determine which simulation types to run:
        # 1. Always run the netlist's native simulation type.
        # 2. Also run any type required by a design goal (e.g. AC for S-params
        #    when the native type is HB).
        native_sim_type: SimulationType = context.native_sim_type
        required_types: set[SimulationType] = set()
        if native_sim_type is not SimulationType.UNKNOWN:
            required_types.add(native_sim_type)

        design_goal_checker = context.design_goal_checker
        required_types.update(design_goal_checker.design_goals)

        # Per-type simulation parameters from the GUI (e.g. sweep range edits)
        sim_params_by_type: dict[SimulationType, dict[str, str]] = context.sim_params_by_type

        netlist_path: str = context.netlist
        # Parsed once; every analysis variant is derived from it in memory.
        netlist = self.simulator.netlist_parser.parse_file(netlist_path)
        name, ext = os.path.splitext(os.path.basename(netlist_path))

        # Start from a clean slate: a result kept from the previous iteration would
        # otherwise be attributed to the current parameters when a simulation fails.
        simulation_results: dict[SimulationType, SimulationResult] = {}
        failed_types: list[SimulationType] = []

        # Run each required simulation type (e.g. AC, HB, TRAN) and store the results in context.
        for sim_type in required_types:
            # Ensure the netlist contains a directive for this simulation type; a
            # netlist that lacks one gets a copy with it injected.
            prepared = self.simulator.prepare_netlist(
                netlist, sim_type, sim_params_by_type.get(sim_type, {})
            )
            prepared_path = netlist_path
            if prepared is not netlist:
                prepared_path = os.path.join(results_dir, f"{name}_{sim_type.name.lower()}{ext}")
                prepared.save(prepared_path)
            # Run the simulation
            sim_result = self.simulator.run_simulation(prepared_path, prepared)

            # Store the results in context for later stages to use.
            if sim_result is None:
                failed_types.append(sim_type)
            else:
                simulation_results[sim_type] = sim_result

        context.simulation_results = simulation_results

        if failed_types:
            logger.warning(
                "No %s result for these parameters; the design goals that need it "
                "are penalised so the optimizer avoids them",
                ", ".join(t.name for t in failed_types),
            )

        return context
