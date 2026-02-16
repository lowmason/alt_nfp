"""Tests for alt_nfp.lookups — industry hierarchy and revision schedules."""

import numpy as np
import pytest

from alt_nfp.lookups import (
    CES_REVISIONS,
    CES_SERIES_MAP,
    INDUSTRY_HIERARCHY,
    QCEW_REVISIONS,
    get_domain_codes,
    get_sector_codes,
    get_supersector_codes,
    sector_to_supersector_idx,
    supersector_to_domain_idx,
)


class TestIndustryHierarchy:
    """Tests for the BLS industry hierarchy."""

    def test_industry_hierarchy_completeness(self):
        """All 10 BLS supersectors present, all sectors map to a valid
        supersector, all supersectors map to a valid domain."""
        df = INDUSTRY_HIERARCHY.collect()

        # 10 unique supersectors
        ss_codes = df['supersector_code'].unique().sort().to_list()
        assert ss_codes == ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']

        # 2 domains
        domains = df['domain_code'].unique().sort().to_list()
        assert domains == ['G', 'S']

        # 18 sectors
        sectors = df['sector_code'].unique().sort().to_list()
        assert len(sectors) == 18

    def test_industry_hierarchy_no_orphans(self):
        """No sector codes without a supersector parent."""
        df = INDUSTRY_HIERARCHY.collect()

        # Every sector has a non-null supersector_code
        assert df['supersector_code'].null_count() == 0

        # Every supersector has a non-null domain_code
        assert df['domain_code'].null_count() == 0

    def test_sector_codes_simplified(self):
        """Sector codes use simplified forms: '31' not '31-33', etc."""
        df = INDUSTRY_HIERARCHY.collect()
        sector_codes = df['sector_code'].to_list()

        # Should not contain range notation
        for code in sector_codes:
            assert '-' not in code, f'Sector code {code!r} uses range notation'

        # Specific checks
        assert '31' in sector_codes  # not '31-33'
        assert '44' in sector_codes  # not '44-45'
        assert '48' in sector_codes  # not '48-49'
        assert '31-33' not in sector_codes
        assert '44-45' not in sector_codes
        assert '48-49' not in sector_codes

    def test_ces_series_id_generation(self):
        """CES series IDs have correct format."""
        # Manufacturing SA
        assert CES_SERIES_MAP[('30', True)] == 'CES3000000001'
        # Manufacturing NSA
        assert CES_SERIES_MAP[('30', False)] == 'CEU3000000001'
        # Total private SA
        assert CES_SERIES_MAP[('05', True)] == 'CES0500000001'
        # Total private NSA
        assert CES_SERIES_MAP[('05', False)] == 'CEU0500000001'

        # All series IDs should match pattern
        for (code, sa), sid in CES_SERIES_MAP.items():
            prefix = 'CES' if sa else 'CEU'
            assert sid.startswith(prefix), f'{sid} should start with {prefix}'
            assert len(sid) == 13, f'{sid} should be 13 chars'
            assert sid.endswith('00000001'), f'{sid} should end with 00000001'

    def test_index_builders_shapes(self):
        """Index builder arrays have correct shapes and value ranges."""
        domains = get_domain_codes()
        ss_codes = get_supersector_codes()
        sec_codes = get_sector_codes()

        n_domains = len(domains)
        n_ss = len(ss_codes)
        n_sec = len(sec_codes)

        # supersector_to_domain_idx
        ss_to_dom = supersector_to_domain_idx()
        assert ss_to_dom.shape == (n_ss,)
        assert ss_to_dom.dtype == np.intp
        assert np.all(ss_to_dom >= 0)
        assert np.all(ss_to_dom < n_domains)

        # sector_to_supersector_idx
        sec_to_ss = sector_to_supersector_idx()
        assert sec_to_ss.shape == (n_sec,)
        assert sec_to_ss.dtype == np.intp
        assert np.all(sec_to_ss >= 0)
        assert np.all(sec_to_ss < n_ss)


class TestRevisionSchedules:
    """Tests for QCEW and CES revision schedules."""

    def test_revision_schedule_counts(self):
        """QCEW: Q1=5, Q2=4, Q3=3, Q4=2 vintages; CES=4 entries."""
        assert len(QCEW_REVISIONS['Q1']) == 5
        assert len(QCEW_REVISIONS['Q2']) == 4
        assert len(QCEW_REVISIONS['Q3']) == 3
        assert len(QCEW_REVISIONS['Q4']) == 2
        assert len(CES_REVISIONS) == 4

    def test_revision_schedule_monotonic_noise(self):
        """Noise multipliers decrease (or stay flat) with revision number."""
        for q_label, specs in QCEW_REVISIONS.items():
            sorted_specs = sorted(specs, key=lambda s: s.revision_number)
            for i in range(1, len(sorted_specs)):
                assert sorted_specs[i].noise_multiplier <= sorted_specs[i - 1].noise_multiplier, (
                    f'QCEW {q_label}: noise should decrease from rev {sorted_specs[i-1].revision_number} '
                    f'to rev {sorted_specs[i].revision_number}'
                )

        # CES: sort by revision_number (excluding benchmark at -1)
        ces_regular = sorted(
            [s for s in CES_REVISIONS if s.revision_number >= 0],
            key=lambda s: s.revision_number,
        )
        for i in range(1, len(ces_regular)):
            assert ces_regular[i].noise_multiplier <= ces_regular[i - 1].noise_multiplier

    def test_revision_schedule_monotonic_lag(self):
        """Lag_months increases with revision number within each schedule."""
        for q_label, specs in QCEW_REVISIONS.items():
            sorted_specs = sorted(specs, key=lambda s: s.revision_number)
            for i in range(1, len(sorted_specs)):
                assert sorted_specs[i].lag_months > sorted_specs[i - 1].lag_months, (
                    f'QCEW {q_label}: lag should increase from rev {sorted_specs[i-1].revision_number} '
                    f'to rev {sorted_specs[i].revision_number}'
                )

        # CES: sort by revision_number (excluding benchmark at -1)
        ces_regular = sorted(
            [s for s in CES_REVISIONS if s.revision_number >= 0],
            key=lambda s: s.revision_number,
        )
        for i in range(1, len(ces_regular)):
            assert ces_regular[i].lag_months > ces_regular[i - 1].lag_months
