# Multi-Period Warehouse Rental LP Optimization

A linear programming model for minimizing warehouse rental cost over a five-month planning horizon while satisfying month-by-month storage requirements.

The implementation uses [PuLP](https://coin-or.github.io/pulp/) and models rental contracts by their start month and duration. A rental started in month `i` for `d` months contributes capacity to every month from `i` through `i + d - 1`.

## Problem Data

### Monthly warehouse requirements

| Month | Required area (m²) |
|---:|---:|
| 1 | 30,000 |
| 2 | 20,000 |
| 3 | 40,000 |
| 4 | 10,000 |
| 5 | 50,000 |

### Rental cost by contract duration

| Duration (months) | Cost (USD/m²) |
|---:|---:|
| 1 | $65 |
| 2 | $100 |
| 3 | $135 |
| 4 | $160 |
| 5 | $190 |

The numeric values are preserved from the source exercise; only the displayed currency label is USD.

## Linear Programming Formulation

Let `x[i,d]` denote the number of square meters rented starting in month `i` for `d` consecutive months.

The objective is

```text
minimize  sum(cost[d] * x[i,d])
```

For every month `t`, all contracts active in that month must provide at least the required warehouse area:

```text
sum(x[i,d] for all i,d with i <= t <= i+d-1) >= requirement[t]
```

with

```text
x[i,d] >= 0
```

The variables are continuous because warehouse area can be modeled in square meters rather than discrete units.

## Verified Optimal Solution

The model produces:

```text
Optimal Rental Plan
-------------------
Rent 30,000 m² in month 1 for 5 months.
Rent 10,000 m² in month 3 for 1 month.
Rent 20,000 m² in month 5 for 1 month.

Minimum Total Cost: $7,650,000.00
```

Monthly coverage is therefore:

| Month | Available area (m²) | Requirement (m²) |
|---:|---:|---:|
| 1 | 30,000 | 30,000 |
| 2 | 30,000 | 20,000 |
| 3 | 40,000 | 40,000 |
| 4 | 30,000 | 10,000 |
| 5 | 50,000 | 50,000 |

The optimum was independently cross-checked with SciPy/HiGHS in addition to the PuLP formulation. Both formulations return an objective value of `$7,650,000` and the same non-zero rental decisions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python warehouse_optimization.py
```

## Tests

```bash
pytest
```

The regression tests verify the expected optimum, monthly feasibility, and consistency between the reported objective and the rental decisions.

## Project Structure

```text
.
├── warehouse_optimization.py
├── tests/
│   └── test_optimization.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## License

This repository is released under a custom **Non-Commercial Software License**. Personal, educational, and academic use is permitted. Commercial use, paid services, resale, or incorporation into commercial products requires prior written permission from the copyright holder.

This is intentionally **not** an OSI-approved open-source license.
