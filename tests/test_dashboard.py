from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_uncaught_exceptions() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dashboard = AppTest.from_file(project_root / "app/dashboard.py", default_timeout=45).run()
    assert not dashboard.exception
    assert [title.value for title in dashboard.title] == ["Retail Demand Control"]
    if dashboard.tabs:
        assert [tab.label for tab in dashboard.tabs] == [
            "Overview",
            "Forecast explorer",
            "Model comparison",
            "Explainability",
            "Inventory",
        ]
    else:
        assert dashboard.error
