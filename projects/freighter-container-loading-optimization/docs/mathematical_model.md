# Mathematical Model

## Sets

Let $I$ be the set of candidate containers.

Let $K$ be the set of incompatible container pairs.

## Parameters

For each container $i \in I$:

- $w_i$: weight of container $i$,
- $q_i$: volume requirement of container $i$,
- $v_i$: economic value of container $i$,
- $p_i \in \{1,2,3\}$: priority class of container $i$, where 1 is the highest priority,
- $r_i \in \{0,1\}$: 1 if container $i$ is refrigerated,
- $h_i \in \{0,1\}$: 1 if container $i$ is hazardous.

Vessel-level parameters:

- $W$: maximum weight capacity,
- $Q$: maximum volume capacity,
- $R$: maximum number of refrigerated containers,
- $H$: maximum number of hazardous containers,
- $P$: minimum number of priority-1 containers,
- $D$: maximum allowed port-starboard weight difference,
- $B$: priority bonus coefficient.

Define $I^{P}$ as the set of containers assigned to the port region and $I^{S}$ as the set of containers assigned to the starboard region.

## Decision variable

$$
x_i = \begin{cases}
1, & \text{if container } i \text{ is loaded}, \\
0, & \text{otherwise}.
\end{cases}
$$

## Objective function

The model maximizes economic value plus a service-priority bonus:

$$
\max Z = \sum_{i \in I} v_i x_i + B\sum_{i \in I}(4-p_i)x_i
$$

The coefficient $(4-p_i)$ assigns larger bonuses to higher-priority containers.

## Constraints

### Weight capacity

$$
\sum_{i \in I} w_i x_i \leq W
$$

### Volume capacity

$$
\sum_{i \in I} q_i x_i \leq Q
$$

### Refrigerated-container capacity

$$
\sum_{i \in I} r_i x_i \leq R
$$

### Hazardous-container limit

$$
\sum_{i \in I} h_i x_i \leq H
$$

### Minimum high-priority service requirement

$$
\sum_{i \in I: p_i=1} x_i \geq P
$$

### Pairwise incompatibility

For each incompatible pair $(i,j) \in K$:

$$
x_i + x_j \leq 1
$$

### Port-starboard balance

The absolute weight difference is linearized using two inequalities:

$$
\sum_{i \in I^{P}} w_i x_i - \sum_{i \in I^{S}} w_i x_i \leq D
$$

$$
\sum_{i \in I^{S}} w_i x_i - \sum_{i \in I^{P}} w_i x_i \leq D
$$

### Binary restriction

$$
x_i \in \{0,1\} \qquad \forall i \in I
$$

## Model class

The formulation is a 0-1 mixed-integer linear programming model. It can also be interpreted as an extended multidimensional knapsack problem with logical and vessel-balance constraints.
