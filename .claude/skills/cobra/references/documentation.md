# COBRA Documentation Workflow

Read the smallest relevant page and name it in the answer. Use source only when
the docs do not settle the question; call out docs/code differences.

| Need | Read first | Source fallback |
| --- | --- | --- |
| Install and prerequisites | `docs/setup.md` | `pyproject.toml` |
| First workflow | `docs/getting-started/quickstart.md` | `examples/main.py` |
| JSON schema and paths | `docs/user-guide/configuration.md` | `src/cobra/configuration/` |
| Python API | `docs/api/core.md`, `docs/user-guide/scripting.md` | `src/cobra/cobra.py` |
| Goals and parameters | README capability tables | `src/cobra/optimizers/design_goal_collection.py`, `design_goal.py` |
| Harmonic Balance | `docs/advanced/harmonic-balance.md` | `src/cobra/spice_sim/hb_spectrum.py` |
| Fine-tuning | `docs/advanced/fine-tuning.md` | `src/cobra/configuration/geometry_loader.py`, `src/cobra/stages/` |
| GUI | `docs/user-guide/gui.md` | `src/cobra/gui/` |
| Errors | `docs/advanced/troubleshooting.md`, `docs/advanced/faq.md` | relevant exception/parser |

Answer with current COBRA terms and a short command or example when useful.
Do not run an optimization just to answer a documentation question.
