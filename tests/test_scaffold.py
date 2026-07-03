"""Scaffold smoke tests: package imports and core dependencies resolve."""


def test_package_imports() -> None:
    import labour_sim

    assert labour_sim.__version__


def test_core_dependencies_import() -> None:
    import mesa
    import numpy
    import pandas
    import pydantic
    import solara

    assert int(mesa.__version__.split(".")[0]) >= 3
    assert numpy and pandas and pydantic and solara
