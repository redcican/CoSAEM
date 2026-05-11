# CoSAEM

**Co**operative **S**urrogate-**A**ssisted **E**volutionary **M**ultitasking for Expensive Constrained Multiobjective Optimization.

## Overview

CoSAEM is a dual-task evolutionary framework that solves an auxiliary unconstrained task alongside the main constrained task. The two tasks cooperate through three mechanisms:

1. **Shared objective surrogates** — GP+RBF ensemble models trained on a shared archive pool data from both tasks.
2. **UPF-guided directional transfer** — Direction vectors from the surrogate-approximated unconstrained Pareto front steer the constrained search toward feasible, well-converged regions.
3. **Population state-driven budget allocation** — A sigmoid-based allocator distributes expensive evaluations between tasks based on feasibility ratio and surrogate accuracy.

