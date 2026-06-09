"""
Stage 3 — Analytical Marts

Input:  outputs/data/enriched.parquet
Output: outputs/data/mart_*.parquet  (one file per mart)

Pre-aggregated tables powering all notebooks and the future dashboard.
All median/percentile stats exclude outlier unit-value rows.
Sum totals (KG, FOB, CIF) always include outlier rows — they represent actual
declared values.

New mart in this version:
  mart_ncm4_annual.parquet — NCM-4 × year, includes HHI and dominant mode/URF.
  Required by notebooks/07_ncm_scoring.ipynb.

Run from project root:
    python pipeline/03_marts.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR

INPUT = OUTPUT_DIR / "enriched.parquet"
OUT   = OUTPUT_DIR


def _in_str() -> str:
    return str(INPUT).replace("\\", "/")


def build_mart(
    con: duckdb.DuckDBPyConnection,
    name: str,
    group_cols: list[str],
    *,
    include_freight: bool = False,
    extra_aggs: str = "",
) -> None:
    """
    Build one mart via DuckDB SQL and write to parquet.

    Sums (KG, FOB, CIF, FRETE) include all rows.
    Median/P10/P90 stats exclude is_outlier_unitval = TRUE rows.
    """
    gc = ", ".join(group_cols)
    group_list = ", ".join(f"g.{c}" for c in group_cols)

    freight_cols = """
            MEDIAN(CASE WHEN NOT is_outlier_unitval THEN FREIGHT_PER_KG  END) AS MEDIAN_FREIGHT_PER_KG,
            MEDIAN(CASE WHEN NOT is_outlier_unitval THEN FREIGHT_PCT_FOB END) AS MEDIAN_FREIGHT_PCT_FOB,
    """ if include_freight else ""

    out_path = str(OUT / f"{name}.parquet").replace("\\", "/")

    con.execute(f"""
        COPY (
            SELECT
                {gc},
                SUM(KG_LIQUIDO)  AS KG_LIQUIDO,
                SUM(VL_FOB)      AS VL_FOB,
                SUM(VL_CIF)      AS VL_CIF,
                SUM(VL_FRETE)    AS VL_FRETE,
                COUNT(*)         AS N_OPS,
                MEDIAN(CASE WHEN NOT is_outlier_unitval THEN UNIT_FOB_PER_KG  END) AS MEDIAN_UNIT_FOB_PER_KG,
                APPROX_QUANTILE(CASE WHEN NOT is_outlier_unitval THEN UNIT_FOB_PER_KG END, 0.10) AS P10_UNIT_FOB_PER_KG,
                APPROX_QUANTILE(CASE WHEN NOT is_outlier_unitval THEN UNIT_FOB_PER_KG END, 0.90) AS P90_UNIT_FOB_PER_KG,
                {freight_cols}
                {extra_aggs}
                1 AS _placeholder
            FROM enriched
            GROUP BY {gc}
            ORDER BY {gc}
        )
        TO '{out_path}'
        (FORMAT PARQUET, CODEC 'SNAPPY')
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    print(f"  {name:45s}: {n:,} rows")


def build_ncm4_annual(con: duckdb.DuckDBPyConnection) -> None:
    """
    mart_ncm4_annual — NCM-4 × year.

    Includes HHI (Herfindahl-Hirschman Index of importers, proxied by URF share),
    dominant transport mode, and dominant port-of-entry (URF).
    Used by notebook 07_ncm_scoring.ipynb.

    Note: SISCOMEX data does not include importer CNPJ at transaction level.
    HHI here is computed on URF (customs station) concentration as a structural
    proxy — a market dominated by 1-2 entry points tends to have concentrated
    supply chains. True importer HHI requires CNPJ-level data.
    """
    out_path = str(OUT / "mart_ncm4_annual.parquet").replace("\\", "/")

    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT
                    CO_POSICAO,
                    CO_ANO,
                    CO_URF,
                    CO_VIA,
                    VL_FOB,
                    KG_LIQUIDO,
                    VL_CIF,
                    VL_FRETE,
                    is_outlier_unitval,
                    UNIT_FOB_PER_KG
                FROM enriched
            ),
            totals AS (
                SELECT
                    CO_POSICAO,
                    CO_ANO,
                    SUM(VL_FOB)      AS total_fob,
                    SUM(KG_LIQUIDO)  AS total_kg,
                    SUM(VL_CIF)      AS total_cif,
                    COUNT(*)         AS n_ops,
                    COUNT(DISTINCT CO_URF) AS n_urfs,
                    MEDIAN(CASE WHEN NOT is_outlier_unitval THEN UNIT_FOB_PER_KG END) AS median_unit_fob,
                    MEDIAN(CASE WHEN NOT is_outlier_unitval THEN VL_FRETE / NULLIF(VL_FOB, 0) END) AS median_freight_pct
                FROM base
                GROUP BY CO_POSICAO, CO_ANO
            ),
            urf_shares AS (
                SELECT
                    CO_POSICAO,
                    CO_ANO,
                    CO_URF,
                    SUM(VL_FOB) AS urf_fob,
                    SUM(SUM(VL_FOB)) OVER (PARTITION BY CO_POSICAO, CO_ANO) AS total_fob_w
                FROM base
                GROUP BY CO_POSICAO, CO_ANO, CO_URF
            ),
            hhi AS (
                SELECT
                    CO_POSICAO,
                    CO_ANO,
                    ROUND(SUM(POWER(urf_fob / NULLIF(total_fob_w, 0) * 100, 2))) AS hhi_urf
                FROM urf_shares
                GROUP BY CO_POSICAO, CO_ANO
            ),
            dominant_mode AS (
                SELECT DISTINCT ON (CO_POSICAO, CO_ANO)
                    CO_POSICAO,
                    CO_ANO,
                    CO_VIA AS modal_dominant
                FROM base
                GROUP BY CO_POSICAO, CO_ANO, CO_VIA
                ORDER BY CO_POSICAO, CO_ANO, SUM(VL_FOB) DESC
            ),
            dominant_urf AS (
                SELECT DISTINCT ON (CO_POSICAO, CO_ANO)
                    CO_POSICAO,
                    CO_ANO,
                    CO_URF AS top_urf
                FROM base
                GROUP BY CO_POSICAO, CO_ANO, CO_URF
                ORDER BY CO_POSICAO, CO_ANO, SUM(VL_FOB) DESC
            )
            SELECT
                t.CO_POSICAO,
                t.CO_ANO,
                t.total_fob,
                t.total_kg,
                t.total_cif,
                t.n_ops,
                t.n_urfs,
                t.median_unit_fob,
                t.median_freight_pct,
                h.hhi_urf,
                m.modal_dominant,
                u.top_urf
            FROM totals      t
            LEFT JOIN hhi          h ON t.CO_POSICAO = h.CO_POSICAO AND t.CO_ANO = h.CO_ANO
            LEFT JOIN dominant_mode m ON t.CO_POSICAO = m.CO_POSICAO AND t.CO_ANO = m.CO_ANO
            LEFT JOIN dominant_urf  u ON t.CO_POSICAO = u.CO_POSICAO AND t.CO_ANO = u.CO_ANO
            ORDER BY t.CO_POSICAO, t.CO_ANO
        )
        TO '{out_path}'
        (FORMAT PARQUET, CODEC 'SNAPPY')
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    print(f"  {'mart_ncm4_annual':45s}: {n:,} rows")


def main() -> None:
    print("=" * 60)
    print("  STAGE 3 — ANALYTICAL MARTS")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run pipeline/02_enrich.py first.")

    con = duckdb.connect()
    in_str = _in_str()
    con.execute(f"CREATE OR REPLACE VIEW enriched AS SELECT * FROM read_parquet('{in_str}')")

    n_rows = con.execute("SELECT COUNT(*) FROM enriched").fetchone()[0]
    sh6_null_pct = 100 * con.execute(
        "SELECT SUM(CASE WHEN CO_SH6 IS NULL THEN 1 ELSE 0 END) FROM enriched"
    ).fetchone()[0] / max(n_rows, 1)

    print(f"\nLoaded {n_rows:,} rows, {len(con.execute('SELECT * FROM enriched LIMIT 0').description)} columns")
    print(f"CO_SH6 null rate: {sh6_null_pct:.1f}%")
    if sh6_null_pct > 5:
        print("  WARNING: >5% of rows have no CO_SH6 — check NCM reference join in 02_enrich.py")

    print("\nBuilding marts ...")

    # 1. Chapter × month — top-level market sizing & trend
    build_mart(con, "mart_chapter_month",
               ["CO_CAPITULO", "CO_ANO", "CO_MES"])

    # 2. NCM-4 × country — origin breakdown, unit value by origin
    build_mart(con, "mart_ncm4_country",
               ["CO_POSICAO", "CO_PAIS", "NO_PAIS_ING"])

    # 3. NCM-4 × URF — trade-lane port-of-entry analysis
    build_mart(con, "mart_ncm4_urf",
               ["CO_POSICAO", "CO_URF", "NO_URF"])

    # 4. NCM-4 × destination state — regional demand by product
    build_mart(con, "mart_ncm4_uf",
               ["CO_POSICAO", "SG_UF_NCM"])

    # 5. HS6 × transport mode — freight intensity by product × mode
    build_mart(con, "mart_sh6_via",
               ["CO_SH6", "CO_VIA", "NO_VIA"],
               include_freight=True)

    # 6. NCM-4 × month — seasonality, trend, YoY
    build_mart(con, "mart_ncm4_month",
               ["CO_POSICAO", "CO_ANO", "CO_MES"])

    # 7. Chapter × country × URF — trade lane flows (Sankey input)
    build_mart(con, "mart_tradeline",
               ["CO_CAPITULO", "CO_PAIS", "NO_PAIS_ING", "CO_URF", "NO_URF", "SG_UF_NCM"])

    # 8. NCM-4 × year — YoY growth, scoring, HHI (NEW)
    build_ncm4_annual(con)

    print(f"\nAll marts written to {OUT}/")
    print("Stage 3 complete.\n")
    print("Verification checklist:")
    print("  [ ] mart_chapter_month.parquet  — Chapter 30 (pharma) in top-10 by VL_FOB")
    print("  [ ] mart_ncm4_country.parquet   — CO_PAIS '160' (China) rows present")
    print("  [ ] mart_sh6_via.parquet        — MEDIAN_FREIGHT_PCT_FOB populated for air (CO_VIA=4)")
    print("  [ ] mart_ncm4_annual.parquet    — one row per CO_POSICAO × CO_ANO, hhi_urf populated")
    print("  [ ] No mart row count > enriched row count (no cartesian explosion)")


if __name__ == "__main__":
    main()
