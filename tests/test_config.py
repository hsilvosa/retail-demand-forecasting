from pathlib import Path

import pytest

from retail_forecasting.config import ProjectConfig, load_config


@pytest.mark.parametrize("profile", ["dev", "full", "test"])
def test_profiles_are_valid(profile: str) -> None:
    config = load_config(profile)
    assert isinstance(config, ProjectConfig)
    assert config.profile == profile
    assert config.data.horizon > 0


def test_unknown_profile_fails() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist", Path("config"))


def test_paths_are_independent_from_current_working_directory() -> None:
    config = load_config("dev")
    assert config.paths.source.is_absolute()
    assert config.paths.source.name == "data"


def test_full_requires_gpu_and_dev_can_fallback() -> None:
    assert load_config("full").models.require_gpu is True
    assert load_config("dev").models.require_gpu is False
