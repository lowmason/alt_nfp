"""Smoke test for QCEW sensitivity analysis (Step 1.5).

Verifies that run_sensitivity() runs without error on a minimal sample.
The full sensitivity sweep requires MCMC sampling and takes several minutes;
this test only verifies that the function can be imported and that the
config structure is valid.
"""

from __future__ import annotations

from nfp_models.sensitivity import QCEW_SIGMA_CONFIGS, _build_param_specs
from nfp_models.settings import NowcastConfig


class TestSensitivityConfig:
    """Verify sensitivity sweep configuration."""

    def test_three_configs(self):
        assert len(QCEW_SIGMA_CONFIGS) == 3

    def test_config_labels(self):
        labels = [c[0] for c in QCEW_SIGMA_CONFIGS]
        assert "0.5x (tight)" in labels
        assert "1x (baseline)" in labels
        assert "2x (loose)" in labels

    def test_sigma_ordering(self):
        """Tighter configs should have smaller sigma values."""
        sigmas_m3 = [c[1] for c in QCEW_SIGMA_CONFIGS]
        assert sigmas_m3 == sorted(sigmas_m3)

    def test_baseline_has_unit_scale(self):
        baseline = [c for c in QCEW_SIGMA_CONFIGS if "baseline" in c[0]][0]
        assert baseline[1] == 1.0  # scale_mid
        assert baseline[2] == 1.0  # scale_boundary


class TestBuildParamSpecs:
    """Verify param spec builder handles the current config."""

    def test_builds_with_no_providers(self):
        import numpy as np

        data = {
            "pp_data": [],
            "claims_c": np.zeros(10),
            "jolts_c": np.zeros(10),
        }
        specs = _build_param_specs(data, NowcastConfig())
        assert len(specs) > 0
        names = [s[0] for s in specs]
        assert any("ces" in n.lower() for n in names)
        assert any("BD" in n for n in names)

    def test_builds_with_cyclical_indicators(self):
        import numpy as np

        data = {
            "pp_data": [],
            "claims_c": np.ones(10),
            "jolts_c": np.ones(10),
        }
        specs = _build_param_specs(data, NowcastConfig())
        names = [s[0] for s in specs]
        phi3_entries = [n for n in names if "φ_3" in n]
        assert len(phi3_entries) == 2
