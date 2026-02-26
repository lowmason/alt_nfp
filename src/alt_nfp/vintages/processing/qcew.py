"""Process the QCEW revisions CSV into ``qcew_revisions.parquet``.

Reads the raw CSV (downloaded by :mod:`~alt_nfp.vintages.download.qcew`),
filters to employment rows, maps area names to FIPS codes, unpivots the
revision columns into long format, joins vintage dates, and writes the result.
"""

from __future__ import annotations

import polars as pl

from alt_nfp.config import DATA_DIR
from alt_nfp.ingest.release_dates.config import VINTAGE_DATES_PATH

OUTPUT_PATH = DATA_DIR / 'raw' / 'qcew_revisions.parquet'
QCEW_CSV_PATH = DATA_DIR / 'raw' / 'qcew' / 'qcew-revisions.csv'

# State name -> FIPS mapping. Built from the geography lookup plus
# the US national entry. Puerto Rico (FIPS '72') is in STATES.
_STATE_FIPS_TO_NAME: dict[str, str] = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut',
    '10': 'Delaware', '11': 'District of Columbia', '12': 'Florida',
    '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky',
    '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
    '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana',
    '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire',
    '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania',
    '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
    '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
    '51': 'Virginia', '53': 'Washington', '54': 'West Virginia',
    '55': 'Wisconsin', '56': 'Wyoming', '72': 'Puerto Rico',
}

STATES_NAME_TO_FIPS: dict[str, str] = {
    'United States': '00',
    **{name: fips for fips, name in _STATE_FIPS_TO_NAME.items()},
}


def main() -> None:
    """Read the QCEW revisions CSV, reshape, join vintage dates, and write Parquet."""
    qcew_1 = (
        pl.read_csv(
            QCEW_CSV_PATH,
            schema_overrides={
                'Year': pl.Utf8,
                'Quarter': pl.Int64,
                'Area': pl.Utf8,
                'Field': pl.Utf8,
                'Initial Value': pl.Utf8,
                'First Revised Value': pl.Utf8,
                'Second Revised Value': pl.Utf8,
                'Third Revised Value': pl.Utf8,
                'Fourth Revised Value': pl.Utf8,
                'Final Value': pl.Utf8,
            },
        )
        .filter(
            pl.col('Area').is_in(STATES_NAME_TO_FIPS.keys()),
            pl.col('Field').str.contains('Employment'),
        )
        .with_columns(
            qtr_date=pl.concat_str(
                pl.lit('12'),
                pl.col('Quarter').mul(3),
                pl.col('Year'),
                separator=' ',
            ).str.to_date(format='%d %m %Y'),
            ref_date=pl.concat_str(
                pl.lit('12'),
                pl.col('Field').str.replace(' Employment', ''),
                pl.col('Year'),
                separator=' ',
            ).str.to_date(format='%d %B %Y'),
            geographic_code=pl.col('Area').replace_strict(
                STATES_NAME_TO_FIPS, default=None,
            ),
        )
        .select(
            qtr_date=pl.col('qtr_date'),
            geographic_type=pl.when(pl.col('geographic_code').eq('00'))
            .then(pl.lit('national'))
            .otherwise(pl.lit('state')),
            geographic_code=pl.col('geographic_code'),
            industry_type=pl.lit('national'),
            industry_code=pl.lit('00'),
            ref_date=pl.col('ref_date'),
            emp_0=pl.col('Initial Value'),
            emp_1=pl.col('First Revised Value'),
            emp_2=pl.col('Second Revised Value'),
            emp_3=pl.col('Third Revised Value'),
            emp_4=pl.col('Fourth Revised Value'),
        )
        .unpivot(
            ['emp_0', 'emp_1', 'emp_2', 'emp_3', 'emp_4'],
            index=[
                'qtr_date', 'geographic_type', 'geographic_code',
                'industry_type', 'industry_code', 'ref_date',
            ],
            value_name='employment',
            variable_name='revision',
        )
        .filter(
            ~pl.col('employment').is_in(['Not yet published', 'Not applicable'])
        )
        .with_columns(
            revision=pl.col('revision')
            .str.replace('emp_', '')
            .cast(pl.UInt8),
            employment=pl.col('employment').cast(pl.Float64).truediv(1000),
        )
        .sort('geographic_code', 'ref_date', 'revision')
    )

    vintage_dates = (
        pl.read_parquet(VINTAGE_DATES_PATH)
        .filter(pl.col('publication').eq('qcew'))
        .select(
            qtr_date=pl.col('ref_date'),
            revision=pl.col('revision').cast(pl.UInt8),
            benchmark_revision=pl.col('benchmark_revision').cast(pl.UInt8),
            vintage_date=pl.col('vintage_date'),
        )
    )

    qcew_2 = qcew_1.join(vintage_dates, on=['qtr_date', 'revision'], how='left')

    output = (
        qcew_2.with_columns(
            source=pl.lit('qcew', pl.Utf8),
            seasonally_adjusted=pl.lit(False, pl.Boolean),
        ).select(
            'source', 'seasonally_adjusted',
            'geographic_type', 'geographic_code',
            'industry_type', 'industry_code',
            'ref_date', 'vintage_date',
            'revision', 'benchmark_revision',
            'employment',
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(OUTPUT_PATH)
    print(f'Wrote {output.height:,} rows to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
