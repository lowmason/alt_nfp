"""Tests for alt_nfp.lookups — industry hierarchy, revision schedules, and geography."""

import numpy as np
import pytest

from alt_nfp.lookups import (
    CES_REVISIONS,
    CES_SERIES_MAP,
    DIVISION_NAMES,
    FIPS_TO_DIVISION,
    FIPS_TO_REGION,
    GEOGRAPHY_HIERARCHY,
    INDUSTRY_HIERARCHY,
    INDUSTRY_MAP,
    IndustryEntry,
    QCEW_REVISIONS,
    REGION_NAMES,
    STATES,
    en_series_id,
    en_series_id_for_state,
    get_domain_codes,
    get_sector_codes,
    get_supersector_codes,
    qcew_to_sector,
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


class TestGeography:
    """Tests for Census Region and Division lookups."""

    def test_states_count(self):
        """52 entries: 50 states + DC + Puerto Rico."""
        assert len(STATES) == 52

    def test_all_fips_mapped_to_region(self):
        """Every FIPS in STATES has a valid region code (1-4)."""
        valid_regions = {'1', '2', '3', '4'}
        for fips in STATES:
            assert fips in FIPS_TO_REGION, f'FIPS {fips} missing from FIPS_TO_REGION'
            assert FIPS_TO_REGION[fips] in valid_regions, (
                f'FIPS {fips} has invalid region {FIPS_TO_REGION[fips]}'
            )

    def test_all_fips_mapped_to_division(self):
        """Every FIPS in STATES has a valid division code (01-09)."""
        valid_divisions = {'01', '02', '03', '04', '05', '06', '07', '08', '09'}
        for fips in STATES:
            assert fips in FIPS_TO_DIVISION, f'FIPS {fips} missing from FIPS_TO_DIVISION'
            assert FIPS_TO_DIVISION[fips] in valid_divisions, (
                f'FIPS {fips} has invalid division {FIPS_TO_DIVISION[fips]}'
            )

    def test_puerto_rico_assignment(self):
        """Puerto Rico (72) → Region 3 (South), Division 05 (South Atlantic)."""
        assert '72' in STATES
        assert FIPS_TO_REGION['72'] == '3'
        assert FIPS_TO_DIVISION['72'] == '05'

    def test_region_names_complete(self):
        """All 4 regions have names."""
        assert len(REGION_NAMES) == 4
        assert set(REGION_NAMES.keys()) == {'1', '2', '3', '4'}

    def test_division_names_complete(self):
        """All 9 divisions have names."""
        assert len(DIVISION_NAMES) == 9
        assert set(DIVISION_NAMES.keys()) == {
            '01', '02', '03', '04', '05', '06', '07', '08', '09'
        }

    def test_geography_hierarchy_lazyframe(self):
        """GEOGRAPHY_HIERARCHY LazyFrame collects with expected shape."""
        df = GEOGRAPHY_HIERARCHY.collect()
        assert df.height == 52
        assert set(df.columns) == {
            'state_fips', 'region_code', 'region_name',
            'division_code', 'division_name',
        }
        assert df['state_fips'].null_count() == 0
        assert df['region_code'].null_count() == 0
        assert df['division_code'].null_count() == 0


class TestIndustryMap:
    """Tests for the CES-QCEW industry cross-mapping."""

    def test_industry_map_has_all_levels(self):
        """INDUSTRY_MAP contains domain, supersector, and sector entries."""
        types = {e.industry_type for e in INDUSTRY_MAP}
        assert types == {'domain', 'supersector', 'sector'}

    def test_industry_map_domain_count(self):
        """5 domain-level entries."""
        domains = [e for e in INDUSTRY_MAP if e.industry_type == 'domain']
        assert len(domains) == 5

    def test_industry_map_supersector_count(self):
        """11 supersector-level entries (10 private + government)."""
        supersectors = [e for e in INDUSTRY_MAP if e.industry_type == 'supersector']
        assert len(supersectors) == 11

    def test_industry_map_sector_count(self):
        """19 sector-level entries."""
        sectors = [e for e in INDUSTRY_MAP if e.industry_type == 'sector']
        assert len(sectors) == 19

    def test_industry_entry_is_frozen(self):
        """IndustryEntry instances are immutable."""
        entry = INDUSTRY_MAP[0]
        with pytest.raises(AttributeError):
            entry.industry_code = 'XX'

    def test_every_entry_has_qcew_naics(self):
        """Every INDUSTRY_MAP entry has a non-empty qcew_naics."""
        for entry in INDUSTRY_MAP:
            assert entry.qcew_naics, f'{entry.industry_code} has empty qcew_naics'

    def test_every_entry_has_ces_code(self):
        """Every INDUSTRY_MAP entry has a 6-digit ces_code."""
        for entry in INDUSTRY_MAP:
            assert len(entry.ces_code) == 6, (
                f'{entry.industry_code} ces_code {entry.ces_code!r} not 6 digits'
            )

    def test_qcew_to_sector_mapping(self):
        """qcew_to_sector() returns expected mappings for known codes."""
        mapping = qcew_to_sector()
        assert mapping['1012'] == '21'   # Mining
        assert mapping['1022'] == '31'   # Manufacturing
        assert mapping['102F'] == '72'   # Accommodation
        # Raw NAICS codes are also mapped
        assert mapping['21'] == '21'
        assert mapping['72'] == '72'

    def test_en_series_id_format(self):
        """EN series IDs have expected prefix and length."""
        # Find the Total Non-Farm domain entry
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid = en_series_id(total_nf)
        assert sid.startswith('EN'), f'EN series ID should start with EN: {sid}'
        assert len(sid) == 17, f'EN series ID should be 17 chars: {sid}'

    def test_en_series_id_for_state(self):
        """State-level EN series ID embeds the state FIPS."""
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid = en_series_id_for_state(total_nf, '26')  # Michigan
        assert '26000' in sid, f'State FIPS 26 should appear in EN series ID: {sid}'

    def test_en_series_id_ownership(self):
        """Private ownership parameter changes EN series ID."""
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid_all = en_series_id(total_nf, ownership='0')
        sid_prv = en_series_id(total_nf, ownership='5')
        assert sid_all != sid_prv, 'Different ownership should produce different IDs'
