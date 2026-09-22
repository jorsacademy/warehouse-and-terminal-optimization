# Heuristics and Metaheuristics Research Series

This file maps repositories that rely on constructive heuristics, local search, adaptive neighborhood search, evolutionary methods, or other derivative-free search strategies. It is an index only: each repository remains independent because the neighborhood structure, representation, and benchmark problem differ.

## Constructive and packing heuristics

- `warehouse-order-packing-heuristics` — shelf and guillotine heuristics for two-dimensional packing.
- `bin-packing-optimization-python` — exact MILP plus First Fit Decreasing comparison.
- `bin-packing-milp-pulp-visualization` — exact one-dimensional bin-packing baseline with visualization.

## Local search and adaptive neighborhood search

- `alns-vehicle-routing` — Adaptive Large Neighborhood Search for vehicle routing.
- `time-dependent-vehicle-routing-alns-python` — ALNS under time-dependent travel conditions.
- `neural-large-neighborhood-search-cvrp` — learned neighborhood guidance for CVRP.
- `neural-large-neighborhood-search-job-shop-scheduling-pytorch` — learned neighborhood search for job-shop scheduling.

## Tabu and evolutionary search

- `capacitated-facility-location-tabu-search-julia` — tabu search for capacitated facility location.
- `paint-shop-scheduling-genetic-algorithm` — genetic algorithm for paint-shop scheduling.
- `flexible-manufacturing-scheduling-genetic-algorithm` — GA for flexible-manufacturing scheduling.
- `airline-crew-workforce-optimization-ga` — GA for crew/workforce optimization.
- `multimodal-distribution-network-genetic-algorithm` — GA for distribution-network decisions.
- `energy-aware-production-scheduling-ga-java` — GA for energy-aware scheduling.
- `multi-objective-cvrp-nsga2-python` — NSGA-II for multi-objective routing.

## General black-box search

- `nevergrad-black-box-policy-optimization` — derivative-free black-box policy optimization.
- `hyperopt-inventory-policy-optimization` — hyperparameter/policy search for inventory control.
- `sambo-sequential-model-based-optimization` — sequential model-based black-box optimization.
- `smac3-simulation-based-optimization` — model-based optimization for expensive/simulation objectives.

## Why these repositories stay separate

The term `heuristic` covers fundamentally different search mechanisms: greedy construction, local search, tabu search, ALNS, genetic algorithms, NSGA-II, neural neighborhood selection, and black-box model-based search. Those differences are central to the research question, so these projects should be compared and cross-linked rather than collapsed into one implementation.
