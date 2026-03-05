"""
Deterministic eval matrix helpers for H3 coverage.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


BucketKey = Tuple[int, int]


@dataclass(frozen=True)
class EvalMatrixPlanEntry:
    """One deterministic eval episode assignment."""

    scenario_id: str
    peer_count: int
    source_rank_ahead: int


def parse_peer_count_list(raw_value: str) -> List[int]:
    """
    Parse comma-separated peer counts into a stable unique list.
    """
    if raw_value is None:
        raise ValueError("peer count list cannot be None")

    values: List[int] = []
    seen = set()
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        count = int(stripped)
        if count <= 0:
            raise ValueError("peer counts must be >= 1")
        if count not in seen:
            values.append(count)
            seen.add(count)

    if not values:
        raise ValueError("peer count list is empty")

    return values


def bucket_label(peer_count: int, source_rank_ahead: int) -> str:
    return f"n{int(peer_count)}_rank{int(source_rank_ahead)}"


def build_deterministic_eval_plan(
    eval_scenario_ids: Sequence[str],
    peer_counts_by_scenario: Mapping[str, int],
    required_peer_counts: Iterable[int],
    episodes_per_bucket: int,
) -> Tuple[List[EvalMatrixPlanEntry], Dict[BucketKey, int]]:
    """
    Build deterministic episode assignments to satisfy per-bucket minimum counts.
    """
    if episodes_per_bucket <= 0:
        raise ValueError("episodes_per_bucket must be > 0")
    if not eval_scenario_ids:
        raise ValueError("eval_scenario_ids is empty")

    required = []
    seen = set()
    for count in required_peer_counts:
        parsed = int(count)
        if parsed <= 0:
            raise ValueError("required_peer_counts must be >= 1")
        if parsed not in seen:
            required.append(parsed)
            seen.add(parsed)

    if not required:
        raise ValueError("required_peer_counts is empty")

    required_set = set(required)
    present_required_counts = set()

    for scenario_id in eval_scenario_ids:
        if scenario_id not in peer_counts_by_scenario:
            raise ValueError(
                f"peer_counts_by_scenario missing scenario_id: {scenario_id}"
            )
        peer_count = int(peer_counts_by_scenario[scenario_id])
        if peer_count <= 0:
            raise ValueError(
                f"peer count for scenario {scenario_id} must be >= 1, got {peer_count}"
            )
        if peer_count in required_set:
            present_required_counts.add(peer_count)

    missing_required = [
        count for count in required if count not in present_required_counts
    ]
    if missing_required:
        missing_str = ",".join(str(value) for value in missing_required)
        raise ValueError(f"missing required peer counts: {missing_str}")

    target_counts: Dict[BucketKey, int] = {}
    for peer_count in required:
        for source_rank_ahead in range(1, peer_count + 1):
            target_counts[(peer_count, source_rank_ahead)] = episodes_per_bucket

    scenarios_by_peer_count: Dict[int, List[str]] = defaultdict(list)
    for scenario_id in eval_scenario_ids:
        scenarios_by_peer_count[int(peer_counts_by_scenario[scenario_id])].append(scenario_id)

    achieved_counts: Dict[BucketKey, int] = defaultdict(int)
    occurrence_index_by_peer_count: Dict[int, int] = defaultdict(int)
    plan: List[EvalMatrixPlanEntry] = []

    max_iterations = 1_000_000
    iterations = 0
    while True:
        if all(
            achieved_counts.get(bucket, 0) >= expected
            for bucket, expected in target_counts.items()
        ):
            break

        if iterations >= max_iterations:
            raise RuntimeError("Exceeded max_iterations while building eval matrix plan")
        iterations += 1

        scenario_id = str(eval_scenario_ids[len(plan) % len(eval_scenario_ids)])
        peer_count = int(peer_counts_by_scenario[scenario_id])

        scenario_group = scenarios_by_peer_count.get(peer_count, [])
        if not scenario_group:
            raise RuntimeError(
                f"No scenarios found for peer_count={peer_count} while building plan"
            )
        group_size = len(scenario_group)
        occurrence_index = occurrence_index_by_peer_count[peer_count]
        slot_index = occurrence_index % group_size
        cycle_index = occurrence_index // group_size
        # Rotate rank assignments across scenario slots each cycle to avoid
        # binding a specific rank to a single scenario.
        current_rank = ((slot_index + cycle_index) % peer_count) + 1
        occurrence_index_by_peer_count[peer_count] += 1

        plan.append(
            EvalMatrixPlanEntry(
                scenario_id=scenario_id,
                peer_count=peer_count,
                source_rank_ahead=current_rank,
            )
        )

        bucket = (peer_count, current_rank)
        if bucket in target_counts and achieved_counts[bucket] < target_counts[bucket]:
            achieved_counts[bucket] += 1

    return plan, target_counts


def summarize_deterministic_eval_coverage(
    episodes: Sequence[dict],
    target_counts: Mapping[BucketKey, int],
) -> dict:
    """
    Compute deterministic eval matrix bucket coverage from episode details.
    """
    observed_counts: Dict[BucketKey, int] = defaultdict(int)
    injection_attempted = 0
    injection_succeeded = 0
    injection_failed = 0
    injection_failed_reasons: Dict[str, int] = defaultdict(int)

    for episode in episodes:
        if episode.get("hazard_injection_attempted", False):
            injection_attempted += 1
            failed_reason = episode.get("hazard_injection_failed_reason")
            if failed_reason:
                injection_failed += 1
                injection_failed_reasons[str(failed_reason)] += 1
            else:
                injection_succeeded += 1

        if episode.get("hazard_step") is None:
            continue
        peer_count = episode.get("peer_count")
        source_rank_ahead = episode.get("hazard_source_rank_ahead")
        if peer_count is None or source_rank_ahead is None:
            continue

        bucket = (int(peer_count), int(source_rank_ahead))
        if bucket in target_counts:
            observed_counts[bucket] += 1

    missing = []
    for bucket, expected in sorted(target_counts.items()):
        observed = observed_counts.get(bucket, 0)
        if observed < expected:
            missing.append(
                {
                    "peer_count": bucket[0],
                    "source_rank_ahead": bucket[1],
                    "expected_episodes": expected,
                    "observed_episodes": observed,
                }
            )

    expected_serialized = {
        bucket_label(peer_count, source_rank_ahead): expected
        for (peer_count, source_rank_ahead), expected in sorted(target_counts.items())
    }
    observed_serialized = {
        bucket_label(peer_count, source_rank_ahead): observed_counts.get(
            (peer_count, source_rank_ahead), 0
        )
        for (peer_count, source_rank_ahead) in sorted(target_counts)
    }

    return {
        "coverage_ok": len(missing) == 0,
        "expected_counts": expected_serialized,
        "observed_counts": observed_serialized,
        "missing_buckets": missing,
        "injection_attempted": injection_attempted,
        "injection_succeeded": injection_succeeded,
        "injection_failed": injection_failed,
        "injection_failed_reasons": dict(injection_failed_reasons),
    }
