# Fine-Tuning

Fine-tuning is an optional phase after surrogate-based optimization.

## Purpose

Surrogate optimization is fast. Fine-tuning adds higher-fidelity EM verification and refinement.

## Requirements

- Palace installed and callable.
- ORCA geometry object available in Python workflow.

## Enabling Fine-Tuning

Fine-tuning behavior is configured via `COBRA(...)` constructor arguments and `cobra.run(...)` context.

Key options:

- `palace_fine_tuning_command`
- `fine_tuning_iterations`
- `fine_tuning_optimizer`

## Fine-Tuning Optimizer Modes

- `reuse`: continue using the surrogate-phase optimizer.
- `gradient_descent`: switch to local gradient-descent refinement.

## High-Level Loop

1. Build geometry from current best parameters.
2. Generate/mesh EM model.
3. Run Palace.
4. Feed verified response back to goal checker.
5. Continue until goals are met or iteration budget is exhausted.

!!! warning
    Fine-tuning requires a correctly configured external EM environment and may be significantly slower than surrogate-only optimization.
