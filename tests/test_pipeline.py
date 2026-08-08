import pytest

from retail_forecasting.pipeline import Stage, select_stages


def no_op(*args: object) -> dict[str, object]:
    return {}


def test_stage_range_is_inclusive() -> None:
    stages = [Stage(name, no_op) for name in ("bronze", "silver", "gold")]
    selected = select_stages(stages, "silver", "gold")
    assert [stage.name for stage in selected] == ["silver", "gold"]


def test_reversed_stage_range_fails() -> None:
    stages = [Stage(name, no_op) for name in ("bronze", "silver", "gold")]
    with pytest.raises(ValueError):
        select_stages(stages, "gold", "silver")
