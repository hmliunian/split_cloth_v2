"""Validate episode concurrency and summarize observed task overlap."""
from __future__ import annotations


def worker_count(value: int, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 16:
        raise ValueError(f"{name}_must_be_integer_1_to_16")
    return value


def execution_stats(tasks: list[dict], max_workers: int, elapsed_s: float) -> dict:
    events = [(task[field], change) for task in tasks
              for field, change in (("started_at", 1), ("finished_at", -1))]
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return {"max_workers": max_workers, "peak_parallel_episodes": peak,
            "wall_time_s": elapsed_s, "tasks": tasks}
