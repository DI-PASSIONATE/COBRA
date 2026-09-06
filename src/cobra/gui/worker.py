import threading

from PySide6.QtCore import QThread, Signal

from cobra.configuration.config_runner import ConfiguredRun


class OptimizationWorker(QThread):
    progress = Signal(dict)
    finished = Signal()
    error = Signal(str)
    ask_continue = Signal(int)

    def __init__(self, configured_run: ConfiguredRun):
        super().__init__()
        self.configured_run = configured_run
        # Runtime knob: the ask-continue dialog raises this mid-run, and the
        # callback pushes the new value into the running loop via the context.
        self.max_iterations = configured_run.configuration.max_iterations
        self.stop_requested = False
        self.paused = False
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.prev_network = None

    def run(self):
        try:
            def optimization_callback(context):
                if self.stop_requested:
                    return False

                if self.paused:
                    self.resume_event.wait()
                    if self.stop_requested:
                        return False

                # Handle prev_network logic for plotting
                context["prev_network"] = self.prev_network

                # Emit progress
                self.progress.emit(context)

                # Check if we reached max iterations and ask to continue
                if context["iteration"] >= context["max_iterations"] and not context["goal_achieved"]:
                    self.ask_continue.emit(context["max_iterations"])
                    context["max_iterations"] = self.max_iterations

                # Update prev_network for next iteration
                sim_results = context.get("simulation_results") or {}
                self.prev_network = next((r.network for r in sim_results.values() if r.network is not None), None)

                return True

            # Same entry point a headless `cobra run CONFIG` takes.
            self.configured_run.run(optimization_callback)

            self.finished.emit()

        except Exception as e:  # noqa: BLE001 - worker thread boundary: failures are forwarded via the error signal
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self.stop_requested = True
        self.resume_event.set()

    def pause(self):
        self.paused = True
        self.resume_event.clear()

    def resume(self):
        self.paused = False
        self.resume_event.set()
