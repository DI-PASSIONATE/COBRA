import copy
import logging
import threading

from PySide6.QtCore import QThread, Signal

from cobra.configuration.config_runner import ConfiguredRun

logger = logging.getLogger(__name__)


class OptimizationWorker(QThread):
    progress = Signal(object)
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

    def run(self):
        try:
            def optimization_callback(context):
                if self.stop_requested:
                    return False

                if self.paused:
                    self.resume_event.wait()
                    if self.stop_requested:
                        return False

                # Emit progress. This signal crosses into the GUI thread, so the
                # slot runs after this callback returns — by which time the
                # optimization loop may have merged a later trial into the same
                # context. Hand over a snapshot so the display cannot show a mix
                # of two trials; the control flow below stays on the live object.
                self.progress.emit(copy.copy(context))

                # Check if we reached max iterations and ask to continue
                if context.iteration >= context.max_iterations and not context.goal_achieved:
                    self.ask_continue.emit(context.max_iterations)
                    context.max_iterations = self.max_iterations

                return True

            # Same entry point a headless `cobra run CONFIG` takes.
            self.configured_run.run(optimization_callback)

            self.finished.emit()

        except Exception as e:  # worker thread boundary: failures are forwarded via the error signal
            logger.exception("The optimization run failed")
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
