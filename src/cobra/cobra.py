import json
import logging
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

import tqdm

from cobra.configuration.configuration import (
    DEFAULT_PALACE_PROCESSES,
    ConfigurationError,
)
from cobra.configuration.setting import CobraSetting
from cobra.optimization_context import OptimizationContext
from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import (
    BaseOptimizer,
    OptimizationProperty,
    OptimizationType,
    netlist_unit,
)
from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages import (
    CircuitSimulationStage,
    EMFineTuningStage,
    EMSurrogateStage,
    OptimizerStage,
)

if TYPE_CHECKING:
    import skrf as rf

    from cobra.configuration import RunConfiguration

logger = logging.getLogger(__name__)


class COBRA:
    """
    COBRA - A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer

    Can be initialized with:
    Pass a NetlistParser and component_onnx_mapping dict
       >>> parser = XyceNetlistParser().from_file("netlist.cir")
       >>> cobra = COBRA(netlist_parser=parser,
       ...               component_onnx_mapping={"X1": "model.onnx"})
    """

    # Settings for run() and __init__ parameters exposed in the GUI.
    _settings: ClassVar[list[CobraSetting]] = [
        CobraSetting(
            name="max_iterations",
            dtype=int,
            default=500,
            description=(
                "Maximum number of surrogate-model optimisation iterations.\n"
                "The loop exits early once all design goals are satisfied."
            ),
        ),
        CobraSetting(
            name="parallel_trials",
            dtype=int,
            default=1,
            description=(
                "Number of optimisation trials evaluated at the same time.\n"
                "Each trial runs its own single-threaded simulation in its own\n"
                "directory, which is usually faster than one MPI-parallel Xyce run.\n"
                "Requires an optimizer that can suggest a trial before the previous\n"
                "one reported back (Optuna can; gradient descent cannot)."
            ),
        ),
        CobraSetting(
            name="fine_tuning_iterations",
            dtype=int,
            default=3,
            description=(
                "Number of EM fine-tuning iterations performed with Palace after the\n"
                "surrogate optimisation loop. Only used when fine-tuning is enabled."
            ),
        ),
        CobraSetting(
            name="palace_fine_tuning_processes",
            dtype=int,
            default=DEFAULT_PALACE_PROCESSES,
            description=(
                "Number of MPI ranks Palace uses per EM fine-tuning simulation.\n"
                "Defaults to the number of cores available on this machine."
            ),
        ),
        CobraSetting(
            name="palace_fine_tuning_command",
            dtype=str,
            default="palace",
            description=(
                "Shell command used to invoke Palace for EM fine-tuning.\n"
                "Can be a plain command name (if on PATH) or an absolute executable path."
            ),
        ),
    ]

    def __init__(
        self,
        netlist_parser: BaseNetlistParser,
        component_onnx_mapping: dict[str, str],
        optimizer: BaseOptimizer | None = None,
        circuit_simulator: BaseSimulator | None = None,
        palace_fine_tuning_command: str | None = None,
        palace_fine_tuning_processes: int = DEFAULT_PALACE_PROCESSES,
        fine_tuning_iterations: int = 3,
        fine_tuning_optimizer: BaseOptimizer | str | None = "reuse",
    ):
        # Validate initialization arguments
        if not isinstance(netlist_parser, BaseNetlistParser):
            raise TypeError("netlist_parser must be an instance of BaseNetlistParser")

        components = netlist_parser.components
        if component_onnx_mapping is None:
            component_onnx_mapping = {}

        # Validate that all components that have models are present in the netlist
        missing_components = set(component_onnx_mapping.keys()) - set(components.keys())
        if missing_components:
            raise ValueError(
                f"component_onnx_mapping references unknown components: {missing_components}. "
                f"Components found in netlist: {set(components.keys())}"
            )

        self.netlist_parser = netlist_parser
        self.component_onnx_mapping = component_onnx_mapping

        # Only create a surrogate stage when there are components with models
        if components and component_onnx_mapping:
            self.em_surrogate_stage = EMSurrogateStage(
                em_surrogate_model=[component_onnx_mapping[comp] for comp in components if comp in component_onnx_mapping],
                component_names=[comp for comp in components if comp in component_onnx_mapping]
            )
        else:
            self.em_surrogate_stage = None

        self.optimizer_stage = OptimizerStage(optimizer if optimizer is not None else OptunaOptimizer())
        self.circuit_simulation_stage = CircuitSimulationStage(
            circuit_simulator if circuit_simulator is not None else XyceSimulator()
        )
        self.em_fine_tuning_stage = (
            EMFineTuningStage(palace_fine_tuning_command, palace_fine_tuning_processes)
            if palace_fine_tuning_command
            else None
        )
        self.fine_tuning_iterations = fine_tuning_iterations
        self.fine_tuning_optimizer = fine_tuning_optimizer

    def _build_fine_tuning_optimizer_stage(self) -> OptimizerStage:
        fine_tuning_optimizer = self.fine_tuning_optimizer

        if fine_tuning_optimizer is None:
            return self.optimizer_stage

        if isinstance(fine_tuning_optimizer, BaseOptimizer):
            return OptimizerStage(fine_tuning_optimizer)

        if isinstance(fine_tuning_optimizer, str):
            normalized_mode = fine_tuning_optimizer.replace("-", "_").strip().lower()
            if normalized_mode in {"reuse", "same", "continue", "surrogate", "surrogate_optimizer"}:
                return self.optimizer_stage
            if normalized_mode in {"gradient_descent", "gradientdescent", "gd"}:
                from cobra.optimizers import GradientDescentOptimizer

                return OptimizerStage(GradientDescentOptimizer())

        raise ValueError(
            "Unsupported fine-tuning optimizer. Choose 'reuse' or 'gradient_descent', or pass a BaseOptimizer instance."
        )

    @staticmethod
    def _validate_parallel_trials(parallel_trials: int) -> None:
        if isinstance(parallel_trials, bool) or not isinstance(parallel_trials, int):
            raise ConfigurationError("parallel_trials must be an integer")
        if parallel_trials < 1:
            raise ConfigurationError(
                f"parallel_trials must be at least 1, got {parallel_trials}"
            )

    def run(self, netlist: str, design_goals: list[DesignGoal], optimization_parameters: list[OptimizationProperty], max_iterations: int = 500, orca_geometries: dict | None = None, callback=None, results_name: str | None = None, sim_params_by_type: dict | None = None, run_configuration: Optional["RunConfiguration"] = None, parallel_trials: int = 1) -> OptimizationContext:
        """
        Run the optimization workflow.

        Parameters:
        - netlist: A path to the netlist file. If a netlist_parser was provided in __init__,
                   it should correspond to this file.
        - design_goals: A list of DesignGoal objects representing the design goals and constraints.
        - optimization_parameters: A list of OptimizationProperty objects representing the parameters
                                   to be optimized, their types, and their ranges.
        - max_iterations: The maximum number of optimization iterations to perform.
        - orca_geometries: Optional dict mapping component names to ORCA geometry objects for EM
                           fine-tuning. Only required for ONNX-based (non-Touchstone) components.
        - callback: An optional callback function that takes the current context as an argument.
                    If the callback returns False, the optimization is stopped.
        - results_name: Optional name for the results folder. If not provided, derives from netlist filename.
        - parallel_trials: How many trials to evaluate at the same time, each in its own
                           working directory. Once a stopping condition is reached the
                           trials already in flight are still finished, so up to
                           parallel_trials - 1 extra evaluations may be performed.

        Returns:
        - The optimized parameters that meet the design goals.
        """
        run_started = time.monotonic()
        self._validate_parallel_trials(parallel_trials)
        optimizer = self.optimizer_stage.optimizer
        if parallel_trials > 1 and not optimizer.supports_parallel_trials:
            raise ConfigurationError(
                f"{type(optimizer).__name__} needs the result of one trial before it can "
                "suggest the next, so parallel_trials must be 1 for it."
            )
        if parallel_trials > 1 and getattr(self.circuit_simulation_stage.simulator, "parallel", False):
            logger.warning(
                "parallel_trials=%d together with a parallel (MPI) simulator oversubscribes "
                "the machine; a single-threaded simulator per trial is usually faster",
                parallel_trials,
            )

        # Create results folder with timestamp and name
        if results_name is None:
            results_name = Path(netlist).stem
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = Path("results") / f"{timestamp}_{results_name}"
        results_dir.mkdir(parents=True, exist_ok=True)

        if run_configuration is not None:
            run_configuration.save(results_dir / "cobra_config.json")

        # Copy original netlist to results directory
        original_netlist_path = Path(netlist)
        netlist_in_results = results_dir / original_netlist_path.name
        shutil.copy(netlist, netlist_in_results)

        # Update netlist path to point to the results directory for all operations
        netlist = str(netlist_in_results)

        netlist_parser = self.netlist_parser

        # Replace the component model names in the netlist to match the vector fitted subcircuits
        for comp_name in self.component_onnx_mapping:
            try:
                netlist_parser.set_model(comp_name, f"{comp_name}_subct")
            except (KeyError, ValueError, NotImplementedError) as e:
                logger.warning("Could not set the subcircuit model for %s: %s", comp_name, e)
        netlist_parser.save(netlist)

        # Every trial renders its own netlist from these lines, so that concurrent
        # trials never share a parser or a file on disk.
        netlist_template = netlist_parser.lines

        design_goal_checker = DesignGoalChecker(design_goals)
        optimizer.initialize(len(design_goals), parallel_trials=parallel_trials)

        context = OptimizationContext(
            netlist=netlist,
            native_sim_type=netlist_parser.simulation_type,
            design_goal_checker=design_goal_checker,
            optimization_parameters=optimization_parameters,
            max_iterations=max_iterations,
            orca_geometries=orca_geometries or {},
            results_dir=str(results_dir),
            sim_params_by_type=sim_params_by_type or {},
        )

        logger.info(
            "Optimizing %s with %d parameter(s) against %d goal(s), up to %d iterations",
            original_netlist_path.name,
            len(optimization_parameters),
            len(design_goals),
            max_iterations,
        )
        logger.info("Writing results to %s", results_dir)

        # Perform optimizer step
        show_progress = logger.isEnabledFor(logging.INFO)
        pbar = tqdm.tqdm(
            total=context.max_iterations,
            desc="COBRA optimization",
            disable=not show_progress,
        )

        # Trials run concurrently, so each gets its own working directory. Only the
        # most recently finished one is kept — the earlier ones have already been
        # reported to the optimizer and the callback.
        trials_dir = results_dir / "trials"
        kept_trial_dir: Path | None = None
        winning_trial: OptimizationContext | None = None
        winning_trial_dir: Path | None = None

        asked = 0
        completed = 0
        stop = False
        in_flight: dict[Future[OptimizationContext], OptimizationContext] = {}

        with ThreadPoolExecutor(max_workers=parallel_trials) as executor:
            while True:
                # Keep the pool topped up. ask() and tell() stay on this thread, so
                # the optimizer never sees concurrent calls.
                while not stop and len(in_flight) < parallel_trials and asked < context.max_iterations:
                    asked += 1
                    trial_context = context.for_trial(trials_dir / f"trial_{asked:04d}")
                    t0 = time.time()
                    self.optimizer_stage.run(trial_context)
                    elapsed = time.time() - t0
                    context.times["optimizer"] += elapsed
                    context.times["total_time"] += elapsed
                    in_flight[
                        executor.submit(self._evaluate_trial, trial_context, netlist_template)
                    ] = trial_context

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    trial_context = in_flight.pop(future)
                    future.result()  # re-raise whatever the worker thread hit
                    completed += 1
                    # Number iterations by completion order so progress never goes backwards.
                    trial_context.iteration = completed
                    context.absorb_trial(trial_context)

                    trial_dir = Path(trial_context.results_dir)
                    if trial_context.goal_achieved and winning_trial is None:
                        winning_trial = trial_context
                        winning_trial_dir = trial_dir

                    logger.debug(
                        "Iteration %d/%d: goals achieved=%s, parameters=%s",
                        completed,
                        context.max_iterations,
                        context.goal_achieved,
                        context.netlist_parameters,
                    )

                    # Callback. Once a stopping condition has been reached the
                    # trials still in flight are drained and told to the optimizer,
                    # but their results are discarded — reporting them would leave
                    # the caller, and the GUI, displaying a trial that lost.
                    if callback and not stop:
                        should_continue = callback(context)
                        if should_continue is False:
                            logger.info("Optimization stopped by the callback at iteration %d", completed)
                            stop = True

                    pbar.update(1)
                    if pbar.total != context.max_iterations:
                        pbar.total = context.max_iterations
                        pbar.refresh()

                    # Tells the optimizer about the current state and saves it to the context for logging
                    self.optimizer_stage.tell(context)

                    if kept_trial_dir is not None and kept_trial_dir != winning_trial_dir:
                        shutil.rmtree(kept_trial_dir, ignore_errors=True)
                    kept_trial_dir = trial_dir

                    # If design goals are achieved, stop submitting; the trials still
                    # in flight are drained by the surrounding loop.
                    if context.goal_achieved:
                        stop = True

                if completed >= context.max_iterations:
                    stop = True

        pbar.close()

        # A trial drained after the successful one leaves its own (failing) result
        # in the context, so reinstate the winner. Its time was already counted.
        if winning_trial is not None:
            context.absorb_trial(winning_trial, include_times=False)

        # If goals not achieved, try to retrieve best parameters from optimizer and use those for final context
        if not context.goal_achieved:
            context = self.re_run_best_parameters(netlist, optimization_parameters, design_goal_checker, netlist_parser, context)
        else:
            # The winning parameters live in a trial directory; put them into the
            # run's own netlist so the saved file, and any fine-tuning, use them.
            netlist_parser.update_parameters(context.netlist_parameters)
            netlist_parser.save(netlist)
            logger.info("Design goals achieved at iteration %s", context.iteration)

        # Save the surrogate model's predicted S-parameters to the results directory for the user
        ntwks: list[rf.Network] = context.predicted_networks
        for i, ntwk in enumerate(ntwks):
            name_suffix = f"_{ntwk.name}" if ntwk.name else f"_{i+1}"
            surrogate_file = results_dir / f"surrogate_s_params{name_suffix}.s{ntwk.nports}p"
            ntwk.write_touchstone(str(surrogate_file))

        if self.em_fine_tuning_stage is not None:
            context.fine_tuning_active = True
            context.fine_tuning_iteration = 0
            context.fine_tuning_total = self.fine_tuning_iterations
            if context.goal_achieved:
                context.fine_tuning_start_iteration = context.iteration

            if callback:
                should_continue = callback(context)
                if should_continue is False:
                    logger.info("EM fine-tuning stopped by the callback before it started")
                    context.wall_time = time.monotonic() - run_started
                    return context

            context = self.fine_tuning(context, callback)

        context.wall_time = time.monotonic() - run_started

        # Save final context to a JSON file for analysis
        context_file = results_dir / "cobra_optimization_context.json"
        with open(context_file, "w") as f:
            json.dump(context.to_json_dict(), f, indent=4, default=str)

        self.log_stage_times(context)
        logger.info("All results saved to %s", results_dir)

        return context

    def _evaluate_trial(
        self, context: OptimizationContext, netlist_template: list[str]
    ) -> OptimizationContext:
        """Render, simulate and score one trial inside its own directory.

        Runs on a worker thread, so it may only touch *context* and objects that
        are safe to share: the netlist parser is built here from
        *netlist_template*, the design goals were copied by
        :meth:`~cobra.optimization_context.OptimizationContext.for_trial`, and the
        stages themselves hold no per-iteration state.
        """
        t1 = time.time()
        parser = type(self.netlist_parser)().from_lines(netlist_template)
        parser.update_parameters(context.netlist_parameters)
        Path(context.results_dir).mkdir(parents=True, exist_ok=True)
        parser.save(context.netlist)  # The circuit simulator reads the netlist from disk

        # Perform EM simulations / s parameter prediction using surrogate model from ORCA
        t2 = time.time()
        if self.em_surrogate_stage is not None:
            self.em_surrogate_stage.run(context)

        # Perform circuit-level simulation
        t3 = time.time()
        self.circuit_simulation_stage.run(context)

        # Check design goals
        t4 = time.time()
        context.design_goal_checker.check_goals(context)
        t5 = time.time()

        # Log times for each stage; the run-wide context sums them up in absorb_trial
        context.times["optimizer"] += t2 - t1
        context.times["em_surrogate"] += t3 - t2
        context.times["circuit_simulation"] += t4 - t3
        context.times["design_goal_checking"] += t5 - t4
        context.times["total_time"] += t5 - t1
        return context

    def re_run_best_parameters(self, netlist, optimization_parameters, design_goal_checker, netlist_parser, context: OptimizationContext) -> OptimizationContext:
        logger.info("Maximum iterations reached without achieving the design goals")

            # If not MOO, retrieve best parameters and update context to reflect them
        if not self.optimizer_stage.optimizer.multi_objective:
            best_params_flat = self.optimizer_stage.optimizer.get_best_parameters()

            # Split best_params_flat into model parameters and netlist parameters based on optimization_parameters list
            model_params = {}
            netlist_params = {}
            by_name = {prop.name: prop for prop in optimization_parameters}

            for prop in optimization_parameters:
                if prop.name in best_params_flat:
                    val = best_params_flat[prop.name]
                    if prop.type == OptimizationType.NETLIST_VARIABLE:
                        # Must match how the optimizers wrote this value during the
                        # run, or the best-parameter netlist is a different circuit.
                        unit = netlist_unit(prop, by_name)
                        netlist_params[prop.name] = f"{val}{unit}"
                    elif prop.type == OptimizationType.MODEL_INPUT:
                        model_params[prop.name] = val

            context.netlist_parameters = netlist_params
            context.model_parameters = model_params

            # Update netlist
            netlist_parser.update_parameters(netlist_params)
            netlist_parser.save(netlist)

            # Rerun simulation to update context with best result
            logger.info("Re-simulating with the best parameters found")
            if self.em_surrogate_stage is not None:
                context = self.em_surrogate_stage.run(context)
            context = self.circuit_simulation_stage.run(context)
            context = design_goal_checker.check_goals(context)

        return context



    def fine_tuning(self, context: OptimizationContext, callback=None) -> OptimizationContext:
        """Perform EM fine-tuning using the EM fine-tuning stage."""
        if self.em_fine_tuning_stage is None:
            raise ValueError("EM fine-tuning stage is not defined. Cannot perform fine-tuning.")

        surrogate_stage = self.em_surrogate_stage
        if surrogate_stage is None:
            raise ValueError(
                "EM fine-tuning requires surrogate components, but none were configured "
                "via component_onnx_mapping. Cannot perform fine-tuning."
            )

        orca_geometries = context.orca_geometries

        # Validate that every ONNX-based component has a geometry
        onnx_components = [
            comp for comp, is_ts in zip(
                surrogate_stage.component_names, surrogate_stage.is_touchstone, strict=True
            )
            if not is_ts
        ]
        missing = [c for c in onnx_components if c not in orca_geometries]
        if missing:
            raise ValueError(
                f"No ORCA geometry provided for component(s): {missing}. "
                "Cannot perform EM fine-tuning without geometry information."
            )

        design_goal_checker: DesignGoalChecker = context.design_goal_checker
        fine_tuning_optimizer_stage = self._build_fine_tuning_optimizer_stage()

        if fine_tuning_optimizer_stage is not self.optimizer_stage:
            fine_tuning_optimizer_stage.optimizer.initialize(len(design_goal_checker.design_goals))

        for iteration in tqdm.tqdm(
            range(self.fine_tuning_iterations),
            desc="COBRA EM fine-tuning",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            context.iteration = iteration + 1
            context.fine_tuning_active = True
            context.fine_tuning_iteration = iteration + 1
            context.fine_tuning_total = self.fine_tuning_iterations

            # Build a name→network map from the previous iteration for .snp components
            prior_networks_by_comp = {
                ntwk.name: ntwk
                for ntwk in context.predicted_networks
                if ntwk.name
            }

            # Run Palace for every ONNX component; carry forward .snp networks unchanged
            assembled_networks = []
            for comp_name, is_ts in zip(
                surrogate_stage.component_names, surrogate_stage.is_touchstone, strict=True
            ):
                if is_ts:
                    ntwk = prior_networks_by_comp.get(comp_name)
                    if ntwk is not None:
                        assembled_networks.append(ntwk)
                else:
                    context = self.em_fine_tuning_stage.run(
                        context,
                        orca_geometry=orca_geometries[comp_name],
                        comp_name=comp_name,
                    )
                    assembled_networks.append(context.predicted_networks[0])

            context.predicted_networks = assembled_networks

            # Perform circuit-level simulation
            context = self.circuit_simulation_stage.run(context)

            # Check design goals
            context = design_goal_checker.check_goals(context)

            if callback:
                should_continue = callback(context)
                if should_continue is False:
                    logger.info("EM fine-tuning stopped by the callback")
                    break

            # If design goals are achieved, break the loop
            if context.goal_achieved:
                logger.info("Design goals achieved after EM fine-tuning iteration %d", iteration + 1)
                break
            logger.info(
                "Design goals not achieved after EM fine-tuning iteration %d; continuing",
                iteration + 1,
            )

            fine_tuning_optimizer_stage.tell(context)
            context = fine_tuning_optimizer_stage.run(context)

        if not context.goal_achieved:
            logger.info(
                "EM fine-tuning finished without achieving the design goals; "
                "returning the best parameters found"
            )
        else:
            logger.info(
                "Design goals achieved and the geometry was verified with an EM simulation"
            )

        return context

    def log_stage_times(self, context: OptimizationContext) -> None:
        """Log how the work of the run was distributed over the stages.

        The shares are of the summed stage times, which exceed the elapsed time
        when trials ran concurrently — so the elapsed time is reported alongside
        them rather than being implied.
        """
        times = context.times
        total_time = times.get("total_time", 0.0)
        if not total_time:
            return

        stages = [
            ("optimizer", "optimizer"),
            ("surrogate", "em_surrogate"),
            ("circuit simulation", "circuit_simulation"),
            ("goal checking", "design_goal_checking"),
        ]
        if self.em_fine_tuning_stage is not None:
            stages.append(("EM fine-tuning", "em_fine_tuning"))

        breakdown = ", ".join(
            f"{label} {times.get(key, 0.0) / total_time * 100:.1f}%" for label, key in stages
        )
        if total_time > context.wall_time * 1.05:
            logger.info(
                "Wall time %.1f s; %.1f s of stage time across concurrent trials: %s",
                context.wall_time,
                total_time,
                breakdown,
            )
        else:
            logger.info("Stage times over %.1f s: %s", total_time, breakdown)
