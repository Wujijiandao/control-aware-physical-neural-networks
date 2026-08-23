# Software design

The code separates five concerns:

1. `viability.py`: digital teacher viability functional used only to define benchmark truth.
2. `dynamics.py`: homeostatic environment and actuator dynamics.
3. `substrates.py`: physics-constrained approximators.
4. `training.py`: ordinary and control-relevant training protocols.
5. `evaluation.py` / `theory.py`: closed-loop metrics and action-preservation bounds.

The final paper should compare multiple physics-constrained substrate families while keeping the homeostatic environment, train/calibration/test splits, perturbation schedule, and statistical protocol fixed.
