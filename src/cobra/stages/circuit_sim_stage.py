import contextlib
import logging
import multiprocessing
import os
import time
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

from cobra.optimizers.design_goal import DesignGoal
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vector_fit import VectorFitError
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.base_stage import COBRABaseStage

if TYPE_CHECKING:
    import skrf as rf

    from cobra.optimization_context import OptimizationContext

logger = logging.getLogger(__name__)


class _ToOwnLogger(logging.Handler):
    """Hand a record from a fitting process to the logger that created it, in this process."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _init_fitting_process(log_queue: "multiprocessing.Queue", level: int) -> None:
    """Send every log record of a fitting process back to the run through *log_queue*.

    The process also keeps to one BLAS thread: by default each one starts a thread
    per core, and the fitting processes together would oversubscribe the machine
    many times over.
    """
    root = logging.getLogger()
    root.handlers = [QueueHandler(log_queue)]
    root.setLevel(level)
    threadpool_limits(limits=1)


def goal_band(goals: list[DesignGoal]) -> tuple[float, float] | None:
    """The frequency span, in Hz, that *goals* evaluate; ``None`` when none names a frequency."""
    lows: list[float] = []
    highs: list[float] = []
    for goal in goals:
        low, high = DesignGoal.str_to_frequency_range(goal.frequency_range)
        if low is not None and high is not None:
            lows.append(low)
            highs.append(high)
    return (min(lows), max(highs)) if lows else None


class CircuitSimulationStage(COBRABaseStage):
    """
    Circuit Simulation Stage — runs one Xyce simulation per required analysis
    type and stores all results in ``context.simulation_results``.

    """

    def __init__(self, simulator: BaseSimulator | None = None):
        self.simulator = simulator if simulator is not None else XyceSimulator("Xyce")
        self._fitting_pool: ProcessPoolExecutor | None = None

    @contextlib.contextmanager
    def fitting_processes(self, processes: int) -> Generator[None]:
        """Fit the surrogates in *processes* worker processes while the block runs.

        Vector fitting is mostly Python, so fits of concurrent trials in threads
        take turns on the GIL and barely overlap. Worker processes fit them in
        parallel. With fewer than two processes the fits stay in the calling
        thread. The workers are spawned, not forked, because the run already has
        threads; a script that runs COBRA this way needs an
        ``if __name__ == "__main__":`` guard.
        """
        if processes < 2:
            yield
            return
        spawn = multiprocessing.get_context("spawn")
        log_queue = spawn.Queue()
        listener = QueueListener(log_queue, _ToOwnLogger())
        listener.start()
        self._fitting_pool = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=spawn,
            initializer=_init_fitting_process,
            initargs=(log_queue, logging.getLogger("cobra").getEffectiveLevel()),
        )
        try:
            yield
        finally:
            self._fitting_pool.shutdown()
            self._fitting_pool = None
            listener.stop()
            log_queue.close()

    def _preprocess(self, ntwk: "rf.Network", name: str) -> None:
        """Preprocess *ntwk* in a fitting process, if :meth:`fitting_processes` started them."""
        if self._fitting_pool is None:
            self.simulator.preprocess_ntwk(ntwk, name=name)
        else:
            self._fitting_pool.submit(self.simulator.preprocess_ntwk, ntwk, name=name).result()

    def run(self, context: "OptimizationContext") -> "OptimizationContext":
        ntwks: list[rf.Network] = context.predicted_networks
        results_dir = context.results_dir

        # Preprocess surrogate models (e.g. vector fitting), timed as its own stage
        fit_started = time.time()
        try:
            for n in ntwks:
                out_name = os.path.join(results_dir, n.name or "cobra_output")
                self._preprocess(n, out_name)
        except VectorFitError as exc:
            logger.warning("%s; the design goals are penalised so the optimizer avoids these parameters", exc)
            context.simulation_results = {}
            return context
        finally:
            context.times["vector_fitting"] += time.time() - fit_started

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
                netlist,
                sim_type,
                sim_params_by_type.get(sim_type, {}),
                goal_band(design_goal_checker.design_goals.get(sim_type, [])),
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
