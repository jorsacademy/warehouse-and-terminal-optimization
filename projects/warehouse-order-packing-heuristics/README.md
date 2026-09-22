# Warehouse Order Packing Heuristics

A small Python project for studying two-dimensional rectangle-packing heuristics in a warehouse fulfillment scenario.

The project uses a fully synthetic dataset representing service-parts orders that must be packed into standardized reusable warehouse totes. The objective is to reduce wasted floor area inside each tote while keeping the implementation simple enough to inspect and extend.

## Scenario

A warehouse prepares outbound service-part orders containing compact electronic modules, cable kits, adapters, control boards, and maintenance components. Each item is approximated as a two-dimensional rectangle. The warehouse uses standardized reusable totes with fixed width and height.

The dataset in this repository is synthetic and was created specifically for this project. It does not represent a real company, customer, product catalog, or operational dataset.

## Implemented methods

- First-Fit Decreasing Height Shelf heuristic
- Best-Fit Shelf heuristic
- Guillotine Best-Area-Fit heuristic

The heuristics are intentionally lightweight. They are not guaranteed to produce globally optimal packings.

## Project structure

```text
warehouse-order-packing-heuristics/
├── README.md
├── LICENSE.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_orders.csv
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── shelf.py
│   ├── guillotine.py
│   ├── metrics.py
│   └── visualization.py
├── examples/
│   └── run_demo.py
└── tests/
    ├── test_shelf.py
    └── test_guillotine.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the demonstration

```bash
python examples/run_demo.py
```

The demonstration loads the synthetic order data, runs the three heuristics, reports utilization metrics, and saves packing visualizations in the `outputs/` directory.

## Metrics

For each method, the demo reports:

- packed area
- used bin count
- average utilization
- total unused area

Utilization is defined as:

```text
packed item area / total available bin area
```

## Design assumptions

This is a two-dimensional packing model. It does not model weight, stacking strength, item fragility, three-dimensional orientation, center of gravity, hazardous-goods constraints, or real carrier rules.

Items may not rotate in the current implementation. This keeps the comparison between heuristics easier to interpret.

## Testing

```bash
pytest
```

The tests cover basic geometric feasibility, oversized-item rejection, non-overlap conditions, and consistency of packing metrics.

## License

This repository is published under a custom non-commercial software license. Educational, personal, and non-commercial research use is permitted. Commercial use is prohibited. See `LICENSE.md` for the full terms.
