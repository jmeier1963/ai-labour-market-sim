"""AI capability frontier and per-task automation quality."""

import numpy as np

from labour_sim.config import CapabilityConfig
from labour_sim.dataset import TaskDef, load_dataset
from labour_sim.sim.capability import CapabilityProcess


def make_process(**overrides) -> CapabilityProcess:
    cfg = CapabilityConfig(**{"noise_sigma": 0.0, **overrides})
    return CapabilityProcess(cfg, np.random.default_rng(0))


def test_capability_grows_toward_ceiling() -> None:
    proc = make_process(initial=0.15, ceiling=0.95, growth_rate=0.05)
    levels = []
    for _ in range(600):
        proc.step()
        levels.append(proc.level)
    assert levels[-1] <= 0.95 + 1e-9
    assert levels[-1] > 0.90
    assert all(b >= a - 1e-12 for a, b in zip(levels, levels[1:])), "noise-free path monotone"


def test_zero_growth_stays_flat() -> None:
    proc = make_process(initial=0.2, growth_rate=0.0)
    for _ in range(50):
        proc.step()
    assert abs(proc.level - 0.2) < 1e-9


def test_task_quality_zero_for_unexposed_and_below_difficulty() -> None:
    proc = make_process(initial=0.5)
    manual = TaskDef(id="m", name="m", exposure=0, difficulty=0.9, augmentation=0.0, source="t")
    hard = TaskDef(id="h", name="h", exposure=2, difficulty=0.8, augmentation=0.2, source="t")
    easy = TaskDef(id="e", name="e", exposure=2, difficulty=0.2, augmentation=0.2, source="t")
    assert proc.task_quality(manual) == 0.0
    assert proc.task_quality(hard) == 0.0
    assert 0.0 < proc.task_quality(easy) <= 1.0


def test_task_quality_monotone_in_capability() -> None:
    ds = load_dataset()
    low, high = make_process(initial=0.3), make_process(initial=0.7)
    for task in ds.tasks.values():
        assert high.task_quality(task) >= low.task_quality(task)


def test_noise_is_seeded_and_bounded() -> None:
    cfg = CapabilityConfig(noise_sigma=0.02, growth_rate=0.03)
    a = CapabilityProcess(cfg, np.random.default_rng(5))
    b = CapabilityProcess(cfg, np.random.default_rng(5))
    for _ in range(100):
        a.step()
        b.step()
        assert a.level == b.level
        assert 0.0 <= a.level <= cfg.ceiling + 1e-9
