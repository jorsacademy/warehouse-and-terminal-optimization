# Freighter Container Loading Optimization

This repository presents a mixed-integer linear programming (MILP) model for selecting containers to load onto a freighter under multiple operational constraints.

The model is designed as an educational Operations Research case study. It extends the classical cargo-loading / knapsack problem by including multiple capacity dimensions and maritime logistics constraints.

## Problem overview

A freighter must select a subset of candidate containers. Each container has a weight, volume requirement, economic value, service priority, cargo type, and assigned vessel region. The objective is to maximize the total economic value of the selected cargo while rewarding high-priority shipments.

The model enforces:

- maximum vessel weight capacity,
- maximum vessel volume capacity,
- refrigerated-container slot availability,
- hazardous-container limits,
- a minimum number of highest-priority containers,
- pairwise incompatibility constraints,
- port-starboard weight-balance limits.

The dataset is deterministic so that the model is fully reproducible for teaching and testing.

## Mathematical structure

Decision variable:

$$
x_i = \begin{cases}
1, & \text{if container } i \text{ is loaded} \\
0, & \text{otherwise}
\end{cases}
$$

Objective function:

$$
\max Z = \sum_{i \in I} v_i x_i + B \sum_{i \in I}(4-p_i)x_i
$$

where $v_i$ is the economic value, $p_i$ is the priority class, and $B$ is a priority bonus coefficient.

See `docs/mathematical_model.md` for the full formulation.

## Verified reference solution

For the included dataset, an independent exhaustive-search validator confirms the optimum:

- objective value: **51,900**
- selected containers: **1, 4, 5, 6, 8, 10, 11, 12**
- total weight: **167 / 170**
- total volume: **242 / 245**
- port-starboard imbalance: **12 / 25**

The PuLP/CBC MILP implementation is expected to reproduce the same optimum.

## Installation

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the optimization

```bash
python src/freighter_optimization.py
```

## Run the independent validator

```bash
python src/exhaustive_validator.py
```

## Run tests

```bash
pytest
```

## Educational use

This repository can be used to teach topics such as:

- 0-1 integer programming,
- multidimensional knapsack models,
- mixed-integer linear programming,
- logical constraints,
- resource-capacity constraints,
- balance constraints,
- independent model validation,
- solver verification.

## License

This project is released under the **JORS Academy Non-Commercial License 1.0**.

Commercial use is prohibited without prior written permission from the copyright holder. See `LICENSE` for the complete terms.
