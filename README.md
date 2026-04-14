# CoSAEM

**Co**operative **S**urrogate-**A**ssisted **E**volutionary **M**ultitasking for Expensive Constrained Multiobjective Optimization.

## Overview

CoSAEM is a dual-task evolutionary framework that solves an auxiliary unconstrained task alongside the main constrained task. The two tasks cooperate through three mechanisms:

1. **Shared objective surrogates** — GP+RBF ensemble models trained on a shared archive pool data from both tasks.
2. **UPF-guided directional transfer** — Direction vectors from the surrogate-approximated unconstrained Pareto front steer the constrained search toward feasible, well-converged regions.
3. **Population state-driven budget allocation** — A sigmoid-based allocator distributes expensive evaluations between tasks based on feasibility ratio and surrogate accuracy.

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Quick Start

```python
from CoSAEM import run_cosam, CoSAEMConfig
from CoSAEM.problems import MW4

problem = MW4(n_var=15)
config = CoSAEMConfig(T_max=300, seed=42)
result = run_cosam(problem, config)

print(f"Found {result.X.shape[0]} non-dominated feasible solutions")
print(f"Used {result.n_evals} function evaluations")
```

## Running Tests

```bash
cd code/
python -m pytest CoSAEM/tests/ -v
```

