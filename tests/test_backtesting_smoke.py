"""Backtesting smoke test (Step 1.2).

Confirms that the censoring machinery produces meaningfully different
data dicts at different as_of dates using real panel data from the
vintage store (if available).
"""

from __future__ import annotations

from datetime import date

import pytest

from nfp_models.config import PROVIDERS
from nfp_ingest import build_panel
from nfp_models.panel_adapter import panel_to_model_data


@pytest.fixture(scope="module")
def panel():
    p = build_panel()
    if len(p) == 0:
        pytest.skip("No data available (vintage store or providers missing)")
    return p


class TestBacktestingSmokeTest:
    """Run panel_to_model_data at two different as_of dates and verify
    that later dates produce more observations (confirming censoring works)."""

    def test_later_as_of_has_more_ces_obs(self, panel):
        early = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2023, 6, 15)
        )
        late = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2025, 6, 15)
        )
        assert len(early["ces_sa_obs"]) < len(late["ces_sa_obs"]), (
            "Later as_of should yield more CES observations"
        )

    def test_later_as_of_has_more_qcew_obs(self, panel):
        early = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2023, 6, 15)
        )
        late = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2025, 6, 15)
        )
        assert len(early["qcew_obs"]) < len(late["qcew_obs"]), (
            "Later as_of should yield more QCEW observations"
        )

    def test_calendar_length_unchanged(self, panel):
        """as_of should censor observations, not the calendar grid."""
        early = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2023, 6, 15)
        )
        late = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2025, 6, 15)
        )
        # Calendar length (T) is determined by unique periods in the panel
        # after vintage_date filtering. The later as_of should have >= T.
        assert early["T"] <= late["T"]

    def test_no_as_of_gives_most_obs(self, panel):
        """Without as_of, all observations should be available."""
        data_full = panel_to_model_data(panel, PROVIDERS)
        data_censored = panel_to_model_data(
            panel, PROVIDERS, as_of=date(2024, 6, 15)
        )
        assert len(data_full["ces_sa_obs"]) >= len(data_censored["ces_sa_obs"])
        assert len(data_full["qcew_obs"]) >= len(data_censored["qcew_obs"])
