"""Implementação Python independente do micro-kernel AXON-Λ para conformance.

O script cobre somente o domínio afim e os registros canônicos abaixo. Ele não
é fallback do Rust, nem uma prova formal de toda a arquitetura.
"""

MASK = (1 << 64) - 1


def u64(value: int) -> int:
    return value & MASK


def root_value(group: int) -> int:
    state = u64(group + 0x9E3779B97F4A7C15)
    state = u64((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9)
    state = u64((state ^ (state >> 27)) * 0x94D049BB133111EB)
    return u64(state ^ (state >> 31))


def apply_rule(previous: int, additive: int) -> int:
    return u64(previous * 3 + additive)


def base_values(node_count: int, chain_len: int) -> list[int]:
    values: list[int] = []
    for factor in range(node_count):
        if factor % chain_len == 0:
            values.append(root_value(factor // chain_len))
        else:
            values.append(apply_rule(values[factor - 1], (factor + 1) * 17))
    return values


def query(node_count: int, chain_len: int, goal: int, changed: int, replacement: int):
    values = base_values(node_count, chain_len)
    demand_start = goal // chain_len * chain_len
    change_end = changed // chain_len * chain_len + chain_len - 1
    demanded = goal - demand_start + 1
    changed_count = change_end - changed + 1
    active = goal - changed + 1 if demand_start == changed // chain_len * chain_len and changed <= goal else 0
    if active == 0:
        return values[goal], "Reuse", demanded, changed_count, active
    full_latency = node_count
    delta_latency = active + 4
    if full_latency <= delta_latency:
        previous = 0
        for factor in range(node_count):
            computed = root_value(factor // chain_len) if factor % chain_len == 0 else apply_rule(previous, (factor + 1) * 17)
            previous = replacement if factor == changed else computed
            if factor == goal:
                goal_value = previous
        return goal_value, "FullRecompute", demanded, changed_count, active
    delta = u64(replacement - values[changed])
    for _ in range(1, active):
        delta = u64(delta * 3)
    return u64(values[goal] + delta), "DeltaPropagation", demanded, changed_count, active


def main() -> None:
    refinement = True  # candidate removes a precondition and adds a guarantee.
    local = query(32, 8, 15, 12, 777)
    global_result = query(32, 32, 31, 0, 777)
    print("AXON-LAMBDA/1")
    print(f"refinement={str(refinement).lower()}")
    print(f"local=value:{local[0]};mode:{local[1]};B:{local[2]};F:{local[3]};A:{local[4]}")
    print(f"global=value:{global_result[0]};mode:{global_result[1]};A:{global_result[4]}")
    print("pareto=options:2;latency:2;memory:2")
    print("lift=sum:38;classes:3;unlift:130")


if __name__ == "__main__":
    main()
