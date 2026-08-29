# Container Terminal Yard Optimization

Mixed-integer linear programming (MILP) example for container terminal yard operations.

The model jointly optimizes:

- yard-block assignment;
- discrete handling time;
- equipment throughput;
- reefer / hazardous / weight compatibility;
- TEU capacity;
- storage, handling, repositioning, and tardiness costs.

## Why this version exists

This project modernizes an older prototype whose service-time formulation was too weak and could silently omit containers that had no compatible block. The revised model validates feasibility before optimization and ties handling times directly to binary scheduling decisions.

## Model

For each container `i`, compatible block `j`, and feasible handling period `t`:

- `x[i,j] = 1` when container `i` is assigned to block `j`;
- `y[i,j,t] = 1` when it is handled in period `t`;
- `delay[i] >= 0` measures tardiness beyond its latest pickup/loading time.

Main constraints:

1. every container is assigned exactly once;
2. TEU assigned to each block does not exceed its capacity;
3. a selected block implies exactly one handling period;
4. handling moves per block/time period respect equipment throughput;
5. handling cannot occur before arrival or the earliest service time;
6. tardiness is computed from the selected handling period.

## Important modeling assumption

Yard capacity is a static planning-horizon approximation. A production-grade terminal model should use time-indexed occupancy so containers consume capacity only while physically present in the yard.

## Install

```bash
python -m pip install -r requirements.txt
```

CBC is normally bundled with PuLP on common desktop Python installations.

## Run

```bash
python container_terminal_optimizer.py
```

## Requirements

- Python 3.10+
- NumPy
- pandas
- Matplotlib
- PuLP
