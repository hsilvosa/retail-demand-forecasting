import numpy as np
import pandas as pd

from retail_forecasting.explainability import analyze_tree_shap


def test_tree_shap_analysis_covers_global_groups_horizons_and_local_rows() -> None:
    frame = pd.DataFrame(
        {
            "series_id": [
                "FOODS_1_001_CA_1",
                "FOODS_1_001_CA_1",
                "HOUSEHOLD_1_001_TX_2",
                "HOUSEHOLD_1_001_TX_2",
            ],
            "horizon": [1, 2, 1, 2],
            "shap_rolling_mean_28": [2.0, -2.0, 1.0, -1.0],
            "shap_target_wday": [0.5, 0.5, -0.5, -0.5],
            "shap_store_id": [0.1, 0.1, 0.1, 0.1],
            "shap_origin_sell_price": [0.0, 0.2, 0.0, 0.2],
        }
    )

    analysis = analyze_tree_shap(frame, top_n=4, local_n=2)

    assert analysis.summary["top_feature"] == "rolling_mean_28"
    assert analysis.summary["sample_rows"] == 4
    assert analysis.summary["series_count"] == 2
    assert analysis.summary["top_feature_horizon_count"] == 2
    assert np.isclose(analysis.group_importance["importance_share"].sum(), 1.0)
    assert set(analysis.group_importance["feature_group"]) == {
        "Demand history",
        "Calendar and horizon",
        "Identifiers",
        "Price",
    }
    assert set(analysis.horizon_importance.loc[lambda data: data["rank"] == 1, "feature"]) == {
        "rolling_mean_28"
    }
    assert set(analysis.segment_importance["segment_type"]) == {
        "state",
        "store",
        "category",
    }
    assert set(analysis.segment_importance.loc[
        analysis.segment_importance["segment_type"] == "store", "segment"
    ]) == {"CA_1", "TX_2"}
    assert len(analysis.local_explanations) == 2
    assert analysis.local_explanations.iloc[0]["top_driver"] == "rolling_mean_28"


def test_tree_shap_analysis_requires_identifiers_and_attributions() -> None:
    try:
        analyze_tree_shap(pd.DataFrame({"series_id": ["a"], "horizon": [1]}))
    except ValueError as error:
        assert "shap_<feature>" in str(error)
    else:
        raise AssertionError("missing SHAP columns should fail")
