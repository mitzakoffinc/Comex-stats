"""
Stage 1 — Clean

Input:  outputs/data/raw_combined.parquet
Output: outputs/data/clean.parquet
        outputs/data/invalid_ncm.parquet

What it does:
  - Validates NCM format (8 digits); quarantines bad rows
  - Adds CO_POSICAO (first 4 chars of CO_NCM) and UNIT_FOB_PER_KG
  - Detects unit-value outliers (IQR per NCM-4 group); flags, never drops
  - Flags zero-weight rows separately (unit-based goods)
  - Marks period completeness dynamically: years with 12 distinct months = complete

Everything runs as DuckDB SQL end-to-end — no pandas round-trip of the full
dataset (only the small config sheet is read with pandas).

Analytical parameters (iqr_multiplier, min_ops_por_grupo) are read from
Config/config.xlsx sheet 'outlier_detection' — edit there, not here.

Run from project root:
    python pipeline/01_clean.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG_XLSX, OUTPUT_DIR

INPUT           = OUTPUT_DIR / "raw_combined.parquet"
OUT_CLEAN       = OUTPUT_DIR / "clean.parquet"
OUT_INVALID_NCM = OUTPUT_DIR / "invalid_ncm.parquet"

# Exactly 8 digits after stripping non-numerics
VALID_NCM = "length(regexp_replace(COALESCE(CO_NCM, ''), '[^0-9]', '', 'g')) = 8"


def _load_outlier_params() -> tuple[float, int]:
    """Read iqr_multiplier and min_ops_por_grupo from config.xlsx."""
    cfg = (
        pd.read_excel(CONFIG_XLSX, sheet_name="outlier_detection")
        .set_index("parametro")["valor"]
    )
    return float(cfg["iqr_multiplier"]), int(cfg["min_ops_por_grupo"])


def main() -> None:
    print("=" * 60)
    print("  STAGE 1 — CLEAN")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run pipeline/00_ingest.py first.")

    iqr_mult, min_ops = _load_outlier_params()
    print(f"\nOutlier params: iqr_multiplier={iqr_mult}, min_ops_por_grupo={min_ops}")

    in_str = str(INPUT).replace("\\", "/")
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet('{in_str}')")

    total_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"\nTotal input rows: {total_rows:,}")

    # -------------------------------------------------------------------------
    # NCM validation — quarantine rows where stripped digits != 8
    # -------------------------------------------------------------------------
    print("\nValidating NCM format ...")

    n_valid, n_invalid = con.execute(f"""
        SELECT
            SUM(CASE WHEN {VALID_NCM} THEN 1 ELSE 0 END),
            SUM(CASE WHEN {VALID_NCM} THEN 0 ELSE 1 END)
        FROM raw
    """).fetchone()
    n_valid, n_invalid = int(n_valid or 0), int(n_invalid or 0)

    print(f"  Valid   : {n_valid:,} ({100*n_valid/total_rows:.2f}%)")
    print(f"  Invalid : {n_invalid:,} ({100*n_invalid/total_rows:.2f}%)")
    if n_invalid > 0:
        samples = con.execute(f"""
            SELECT DISTINCT CO_NCM FROM raw WHERE NOT ({VALID_NCM}) LIMIT 10
        """).fetchall()
        print(f"  Sample bad NCMs: {[r[0] for r in samples]}")

    # -------------------------------------------------------------------------
    # Derived columns on valid set
    # -------------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE VIEW valid_derived AS
        SELECT *,
            LEFT(CO_NCM, 4)                                   AS CO_POSICAO,
            CASE
                WHEN KG_LIQUIDO > 0 THEN VL_FOB / KG_LIQUIDO
                ELSE NULL
            END                                               AS UNIT_FOB_PER_KG,
            (KG_LIQUIDO = 0 OR KG_LIQUIDO IS NULL)           AS is_zero_weight
        FROM raw
        WHERE {VALID_NCM}
    """)

    # -------------------------------------------------------------------------
    # Period completeness — dynamic: years with 12 distinct months = complete
    # -------------------------------------------------------------------------
    print("Detecting complete periods dynamically ...")

    period_rows = con.execute("""
        SELECT CO_ANO, COUNT(DISTINCT CO_MES) AS n_months
        FROM valid_derived
        WHERE CO_ANO IS NOT NULL
        GROUP BY CO_ANO
    """).fetchall()

    max_year = max((r[0] for r in period_rows), default=None)
    complete_years = {r[0] for r in period_rows if r[1] == 12}

    print(f"  Complete years (12 months): {sorted(complete_years)}")
    if max_year and max_year not in complete_years:
        partial_months = next(r[1] for r in period_rows if r[0] == max_year)
        print(f"  Partial year {max_year}: data through month {partial_months} only")

    complete_years_sql = ", ".join(str(y) for y in sorted(complete_years)) or "NULL"

    con.execute(f"""
        CREATE OR REPLACE VIEW valid_with_period AS
        SELECT *,
            (CO_ANO IN ({complete_years_sql})) AS is_complete_period
        FROM valid_derived
    """)

    # -------------------------------------------------------------------------
    # IQR outlier detection per NCM-4 group
    # Groups with < min_ops rows are excluded from outlier flagging
    # -------------------------------------------------------------------------
    print(f"Flagging outlier unit values (IQR × {iqr_mult} per NCM-4) ...")

    con.execute(f"""
        CREATE OR REPLACE VIEW valid_with_outliers AS
        WITH group_stats AS (
            SELECT
                CO_POSICAO,
                COUNT(*)                                                   AS grp_count,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY UNIT_FOB_PER_KG) AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY UNIT_FOB_PER_KG) AS q3
            FROM valid_with_period
            WHERE UNIT_FOB_PER_KG IS NOT NULL AND VL_FOB > 0
            GROUP BY CO_POSICAO
        ),
        with_bounds AS (
            SELECT
                CO_POSICAO,
                q1 - {iqr_mult} * (q3 - q1) AS lo,
                q3 + {iqr_mult} * (q3 - q1) AS hi
            FROM group_stats
            WHERE grp_count >= {min_ops}
        )
        SELECT
            v.*,
            CASE
                WHEN v.UNIT_FOB_PER_KG IS NOT NULL
                     AND v.VL_FOB > 0
                     AND b.CO_POSICAO IS NOT NULL
                     AND (v.UNIT_FOB_PER_KG < b.lo OR v.UNIT_FOB_PER_KG > b.hi)
                THEN TRUE
                ELSE FALSE
            END AS is_outlier_unitval
        FROM valid_with_period v
        LEFT JOIN with_bounds b ON v.CO_POSICAO = b.CO_POSICAO
    """)

    # -------------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------------
    stats = con.execute("""
        SELECT
            SUM(CASE WHEN is_outlier_unitval THEN 1 ELSE 0 END)  AS n_outlier,
            SUM(CASE WHEN is_zero_weight     THEN 1 ELSE 0 END)  AS n_zero_wt,
            SUM(CASE WHEN NOT is_complete_period THEN 1 ELSE 0 END) AS n_incomplete
        FROM valid_with_outliers
    """).fetchone()
    n_outlier, n_zero_wt, n_incomplete = stats

    print(f"\n  Outlier unit-value flags  : {n_outlier:,} ({100*n_outlier/n_valid:.2f}%)")
    print(f"  Zero-weight flags         : {n_zero_wt:,} ({100*n_zero_wt/n_valid:.2f}%)")
    print(f"  Incomplete-period rows    : {n_incomplete:,} ({100*n_incomplete/n_valid:.2f}%)")

    # -------------------------------------------------------------------------
    # Write outputs — straight from DuckDB, no pandas materialization
    # -------------------------------------------------------------------------
    clean_str   = str(OUT_CLEAN).replace("\\", "/")
    invalid_str = str(OUT_INVALID_NCM).replace("\\", "/")

    print(f"\nWriting {OUT_CLEAN.name} ...")
    con.execute(f"""
        COPY (SELECT * FROM valid_with_outliers)
        TO '{clean_str}'
        (FORMAT PARQUET, CODEC 'SNAPPY')
    """)
    size_mb = OUT_CLEAN.stat().st_size / 1e6
    print(f"  {n_valid:,} rows → {size_mb:.1f} MB")

    if n_invalid > 0:
        print(f"Writing {OUT_INVALID_NCM.name} ...")
        con.execute(f"""
            COPY (SELECT * FROM raw WHERE NOT ({VALID_NCM}))
            TO '{invalid_str}'
            (FORMAT PARQUET, CODEC 'SNAPPY')
        """)
        print(f"  {n_invalid:,} rows quarantined")
    else:
        print("  No invalid NCM rows — invalid_ncm.parquet not created")

    print("\nStage 1 complete. Next: pipeline/02_enrich.py")


if __name__ == "__main__":
    main()
