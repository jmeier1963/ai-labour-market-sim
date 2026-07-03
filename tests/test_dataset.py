"""Bundled task/occupation/sector data: schema validity and internal consistency."""

import math

from labour_sim.dataset import load_dataset


def test_dataset_loads() -> None:
    ds = load_dataset()
    assert len(ds.tasks) >= 20
    assert len(ds.occupations) >= 20
    assert len(ds.sectors) == 8


def test_task_fields_in_range() -> None:
    ds = load_dataset()
    for task in ds.tasks.values():
        assert task.exposure in (0, 1, 2)
        assert 0.0 <= task.difficulty <= 1.0
        assert 0.0 <= task.augmentation <= 1.0
        assert task.source, f"task {task.id} missing source"


def test_occupation_weights_reference_known_tasks_and_normalize() -> None:
    ds = load_dataset()
    for occ in ds.occupations.values():
        assert occ.task_weights, occ.id
        for task_id in occ.task_weights:
            assert task_id in ds.tasks, f"{occ.id} references unknown task {task_id}"
        assert math.isclose(sum(occ.task_weights.values()), 1.0, abs_tol=1e-6)


def test_employment_shares_normalized() -> None:
    ds = load_dataset()
    assert math.isclose(sum(o.employment_share for o in ds.occupations.values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(s.employment_share for s in ds.sectors.values()), 1.0, abs_tol=1e-6)


def test_sector_mixes_reference_known_occupations() -> None:
    ds = load_dataset()
    for sector in ds.sectors.values():
        for occ_id in sector.occupation_mix:
            assert occ_id in ds.occupations
        assert math.isclose(sum(sector.occupation_mix.values()), 1.0, abs_tol=1e-6)


def test_occupation_distance_symmetric_and_bounded() -> None:
    ds = load_dataset()
    ids = list(ds.occupations)
    a, b = ids[0], ids[1]
    assert ds.occupation_distance(a, a) < 1e-9
    assert abs(ds.occupation_distance(a, b) - ds.occupation_distance(b, a)) < 1e-12
    assert 0.0 <= ds.occupation_distance(a, b) <= 1.0
