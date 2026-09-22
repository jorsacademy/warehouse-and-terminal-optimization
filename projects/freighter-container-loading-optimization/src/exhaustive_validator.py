import itertools

from freighter_optimization import (
    CONTAINERS,
    INCOMPATIBLE_PAIRS,
    MAX_HAZARDOUS,
    MAX_PORT_STARBOARD_IMBALANCE,
    MAX_REEFER,
    MAX_VOLUME,
    MAX_WEIGHT,
    MIN_PRIORITY_1,
    PRIORITY_BONUS,
)


def exhaustive_optimum() -> dict:
    ids = [c.container_id for c in CONTAINERS]
    by_id = {c.container_id: c for c in CONTAINERS}
    best = None

    for bits in itertools.product([0, 1], repeat=len(ids)):
        x = dict(zip(ids, bits))
        chosen = [by_id[i] for i in ids if x[i] == 1]

        total_weight = sum(c.weight for c in chosen)
        total_volume = sum(c.volume for c in chosen)
        total_reefers = sum(c.reefer for c in chosen)
        total_hazardous = sum(c.hazardous for c in chosen)
        priority_1_count = sum(c.priority == 1 for c in chosen)

        if total_weight > MAX_WEIGHT:
            continue
        if total_volume > MAX_VOLUME:
            continue
        if total_reefers > MAX_REEFER:
            continue
        if total_hazardous > MAX_HAZARDOUS:
            continue
        if priority_1_count < MIN_PRIORITY_1:
            continue
        if any(x[i] + x[j] > 1 for i, j in INCOMPATIBLE_PAIRS):
            continue

        port_weight = sum(c.weight for c in chosen if c.region == "port")
        starboard_weight = sum(c.weight for c in chosen if c.region == "starboard")
        imbalance = abs(port_weight - starboard_weight)

        if imbalance > MAX_PORT_STARBOARD_IMBALANCE:
            continue

        objective = sum(
            c.value + PRIORITY_BONUS * (4 - c.priority)
            for c in chosen
        )

        candidate = {
            "objective": objective,
            "selected": [c.container_id for c in chosen],
            "total_weight": total_weight,
            "total_volume": total_volume,
            "total_reefers": total_reefers,
            "total_hazardous": total_hazardous,
            "priority_1_count": priority_1_count,
            "port_weight": port_weight,
            "starboard_weight": starboard_weight,
            "imbalance": imbalance,
        }

        if best is None or candidate["objective"] > best["objective"]:
            best = candidate

    if best is None:
        raise RuntimeError("No feasible solution exists.")

    return best


def main() -> None:
    result = exhaustive_optimum()
    print(f"Validated objective: {result['objective']}")
    print(f"Selected containers: {result['selected']}")
    print(f"Total weight: {result['total_weight']} / {MAX_WEIGHT}")
    print(f"Total volume: {result['total_volume']} / {MAX_VOLUME}")
    print(f"Port-starboard imbalance: {result['imbalance']} / {MAX_PORT_STARBOARD_IMBALANCE}")


if __name__ == "__main__":
    main()
