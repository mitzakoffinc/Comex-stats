"""
Stage 0 — Ingest & Validate

Reads all IMP_*.csv files from Data/Imports Data/ via DuckDB, profiles each year,
and writes outputs/data/raw_combined.parquet.

Dynamic: drop any correctly-formatted IMP_*.csv into the imports folder and re-run —
no code changes needed.

Run from project root:
    python pipeline/00_ingest.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OUTPUT_DIR,
    STRING_COLS,
    NUMERIC_INT_COLS,
    NUMERIC_FLOAT_COLS,
    discover_import_files,
    import_glob_str,
)

OUTPUT = OUTPUT_DIR / "raw_combined.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIZ_KEY = ["CO_ANO", "CO_MES", "CO_NCM", "CO_PAIS", "SG_UF_NCM", "CO_VIA", "CO_URF"]


def _types_map() -> str:
    """Build a DuckDB struct literal for column type overrides."""
    parts = []
    for col in STRING_COLS:
        parts.append(f"'{col}': 'VARCHAR'")
    for col in NUMERIC_INT_COLS:
        parts.append(f"'{col}': 'BIGINT'")
    for col in NUMERIC_FLOAT_COLS:
        parts.append(f"'{col}': 'DOUBLE'")
    return "{" + ", ".join(parts) + "}"


def _read_expr(glob_str: str) -> str:
    return (
        f"read_csv('{glob_str}', sep=';', quote='\"', encoding='Latin-1', "
        f"types={_types_map()}, ignore_errors=true)"
    )


def main() -> None:
    print("=" * 60)
    print("  STAGE 0 — INGEST & VALIDATE")
    print("=" * 60)

    files = discover_import_files()
    print(f"\nFound {len(files)} import file(s):")
    for f in files:
        print(f"  {f.name}")

    glob_str  = import_glob_str()
    read_expr = _read_expr(glob_str)

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM {read_expr}")

    # --- Per-year stats ---
    print("\nPer-year profile:")

    year_rows = con.execute("""
        SELECT
            CO_ANO                                           AS year,
            COUNT(*)                                         AS rows,
            COUNT(DISTINCT CO_MES)                           AS n_months,
            MIN(CO_MES)                                      AS mes_min,
            MAX(CO_MES)                                      AS mes_max,
            COUNT(DISTINCT CO_NCM)                           AS unique_ncm,
            COUNT(DISTINCT CO_PAIS)                          AS unique_pais,
            COUNT(DISTINCT CO_URF)                           AS unique_urf,
            COUNT(DISTINCT SG_UF_NCM)                        AS unique_uf,
            SUM(CASE WHEN VL_FOB     < 0 THEN 1 ELSE 0 END) AS neg_fob,
            SUM(CASE WHEN KG_LIQUIDO < 0 THEN 1 ELSE 0 END) AS neg_kg,
            SUM(CASE WHEN VL_FRETE   < 0 THEN 1 ELSE 0 END) AS neg_frete,
            SUM(CASE WHEN VL_FOB     = 0 THEN 1 ELSE 0 END) AS zero_fob,
            SUM(CASE WHEN KG_LIQUIDO = 0 THEN 1 ELSE 0 END) AS zero_kg
        FROM raw
        GROUP BY CO_ANO
        ORDER BY CO_ANO
    """).fetchall()

    max_year = max((r[0] for r in year_rows if r[0] is not None), default=None)

    for (year, rows, n_months, mes_min, mes_max,
         u_ncm, u_pais, u_urf, u_uf,
         neg_fob, neg_kg, neg_frete, zero_fob, zero_kg) in year_rows:

        print(f"\n{'='*60}")
        print(f"  YEAR: {year}")
        print(f"{'='*60}")
        print(f"  Rows              : {rows:,}")
        print(f"  Months found      : {n_months}  (range {mes_min}–{mes_max})")
        if year == max_year and n_months < 12:
            print(f"  *** PARTIAL YEAR: data through month {mes_max} only.")
            print(f"      Do NOT use raw {year} totals for YoY without annualizing.")
        print(f"  Unique NCMs       : {u_ncm:,}")
        print(f"  Unique countries  : {u_pais:,}")
        print(f"  Unique URFs       : {u_urf:,}")
        print(f"  Unique UF states  : {u_uf:,}")
        print(f"  Anomalies:")
        print(f"    Negative VL_FOB     : {neg_fob:,}")
        print(f"    Negative KG_LIQUIDO : {neg_kg:,}")
        print(f"    Negative VL_FRETE   : {neg_frete:,}")
        print(f"    Zero VL_FOB         : {zero_fob:,}")
        print(f"    Zero KG_LIQUIDO     : {zero_kg:,}")

    # --- Combined null rates ---
    print(f"\n{'='*60}")
    print("  COMBINED DATASET — NULL RATES")
    print(f"{'='*60}")

    null_cols    = ["CO_NCM", "CO_PAIS", "CO_URF", "KG_LIQUIDO", "VL_FOB", "VL_FRETE", "VL_SEGURO"]
    null_selects = ", ".join(
        f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS n_{c}" for c in null_cols
    )
    null_row = con.execute(
        f"SELECT COUNT(*) AS total, {null_selects} FROM raw"
    ).fetchone()

    total = null_row[0]
    print(f"  Total rows        : {total:,}")
    for i, col in enumerate(null_cols):
        n   = null_row[i + 1]
        pct = 100 * n / max(total, 1)
        print(f"    {col:<13}: {n:,} ({pct:.1f}%)")

    # --- Duplicate business-key check ---
    key_cols  = ", ".join(BIZ_KEY)
    dup_extra = con.execute(f"""
        SELECT COALESCE(SUM(cnt - 1), 0)
        FROM (
            SELECT COUNT(*) AS cnt
            FROM raw
            GROUP BY {key_cols}
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    # --- Bad NCM format (must be exactly 8 digits) ---
    bad_ncm = con.execute("""
        SELECT COUNT(*) FROM raw
        WHERE length(regexp_replace(COALESCE(CO_NCM, ''), '[^0-9]', '', 'g')) != 8
    """).fetchone()[0]

    print(f"\n  Dup biz-key rows  : {dup_extra:,}")
    print(f"  Bad NCM format    : {bad_ncm:,}")

    # --- Write parquet ---
    out_str = str(OUTPUT).replace("\\", "/")
    print(f"\nWriting {OUTPUT.name} ...")
    con.execute(f"""
        COPY (SELECT * FROM raw)
        TO '{out_str}'
        (FORMAT PARQUET, CODEC 'SNAPPY')
    """)

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"  Done. {total:,} rows, {size_mb:.1f} MB")
    print("\nStage 0 complete. Next: pipeline/01_clean.py")


if __name__ == "__main__":
    main()
