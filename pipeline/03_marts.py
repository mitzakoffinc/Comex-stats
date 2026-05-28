"""
Stage 3 — Analytical Marts

Input:  outputs/data/enriched.parquet
Output: outputs/data/mart_*.parquet  (one file per mart)

Pre-aggregated tables powering all notebooks and the future dashboard.
All marts exclude outlier unit-value rows from median/percentile calculations
(but include them in kg/FOB/CIF totals — totals represent actual declared values).

Run from project root:
    python pipeline/03_marts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT  = PROJECT_ROOT / "outputs" / "data" / "enriched.parquet"
OUT    = PROJECT_ROOT / "outputs" / "data"

AGGS_BASE = {
    "KG_LIQUIDO": "sum",
    "VL_FOB":     "sum",
    "VL_CIF":     "sum",
    "VL_FRETE":   "sum",
}


def p10(s: pd.Series) -> float:
    return float(s.quantile(0.10)) if s.notna().any() else np.nan

def p90(s: pd.Series) -> float:
    return float(s.quantile(0.90)) if s.notna().any() else np.nan


def unit_price_aggs(group_df: pd.DataFrame, price_col: str = "UNIT_FOB_PER_KG") -> dict:
    """Return median/P10/P90 of unit price, excluding outlier-flagged rows."""
    clean = group_df.loc[~group_df["is_outlier_unitval"], price_col].dropna()
    return {
        f"MEDIAN_{price_col}": float(clean.median()) if len(clean) else np.nan,
        f"P10_{price_col}":    p10(clean),
        f"P90_{price_col}":    p90(clean),
    }


def freight_aggs(group_df: pd.DataFrame) -> dict:
    frt_kg  = group_df["FREIGHT_PER_KG"].dropna()
    frt_pct = group_df["FREIGHT_PCT_FOB"].dropna()
    return {
        "MEDIAN_FREIGHT_PER_KG":  float(frt_kg.median())  if len(frt_kg)  else np.nan,
        "MEDIAN_FREIGHT_PCT_FOB": float(frt_pct.median()) if len(frt_pct) else np.nan,
    }


def ops_count(group_df: pd.DataFrame) -> int:
    return len(group_df)


def build_mart(
    df: pd.DataFrame,
    group_cols: list[str],
    name: str,
    include_freight: bool = False,
) -> pd.DataFrame:
    """Generic mart builder: sum numerics, compute unit-price stats per group."""
    groups = df.groupby(group_cols, observed=True)

    # Sum aggregations
    agg_df = groups.agg(AGGS_BASE).reset_index()
    agg_df.columns = group_cols + list(AGGS_BASE.keys())
    agg_df["N_OPS"] = groups.size().values

    # Unit price stats (outliers excluded from price calcs)
    price_rows = []
    for key, grp in groups:
        row = {c: v for c, v in zip(group_cols, (key if isinstance(key, tuple) else (key,)))}
        row.update(unit_price_aggs(grp))
        if include_freight:
            row.update(freight_aggs(grp))
        price_rows.append(row)

    price_df = pd.DataFrame(price_rows)
    result = agg_df.merge(price_df, on=group_cols, how="left")

    print(f"  {name:40s}: {len(result):,} rows")
    return result


def write(df: pd.DataFrame, filename: str) -> None:
    path = OUT / filename
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="snappy",
    )


# ---------------------------------------------------------------------------
# Mart definitions
# ---------------------------------------------------------------------------

def build_all(df: pd.DataFrame) -> None:
    # 1. Chapter × month — top-level market sizing & trend
    m = build_mart(
        df,
        group_cols=["CO_CAPITULO", "CO_ANO", "CO_MES"],
        name="mart_chapter_month",
    )
    write(m, "mart_chapter_month.parquet")

    # 2. NCM-4 × country — origin breakdown, unit value by origin
    m = build_mart(
        df,
        group_cols=["CO_POSICAO", "CO_PAIS", "NO_PAIS_ING"],
        name="mart_ncm4_country",
    )
    write(m, "mart_ncm4_country.parquet")

    # 3. NCM-4 × URF — trade-lane port-of-entry analysis
    m = build_mart(
        df,
        group_cols=["CO_POSICAO", "CO_URF", "NO_URF"],
        name="mart_ncm4_urf",
    )
    write(m, "mart_ncm4_urf.parquet")

    # 4. NCM-4 × destination state — regional demand by product
    m = build_mart(
        df,
        group_cols=["CO_POSICAO", "SG_UF_NCM"],
        name="mart_ncm4_uf",
    )
    write(m, "mart_ncm4_uf.parquet")

    # 5. HS6 × transport mode — freight intensity by product × mode
    m = build_mart(
        df,
        group_cols=["CO_SH6", "CO_VIA", "NO_VIA"],
        name="mart_sh6_via (freight)",
        include_freight=True,
    )
    write(m, "mart_sh6_via.parquet")

    # 6. NCM-4 × month — seasonality, trend, YoY
    m = build_mart(
        df,
        group_cols=["CO_POSICAO", "CO_ANO", "CO_MES"],
        name="mart_ncm4_month",
    )
    write(m, "mart_ncm4_month.parquet")

    # 7. Chapter × country × URF — trade lane flows (Sankey input)
    m = build_mart(
        df,
        group_cols=["CO_CAPITULO", "CO_PAIS", "NO_PAIS_ING", "CO_URF", "NO_URF", "SG_UF_NCM"],
        name="mart_tradeline",
    )
    write(m, "mart_tradeline.parquet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  STAGE 3 — ANALYTICAL MARTS")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run pipeline/02_enrich.py first.")

    print(f"\nReading {INPUT.name} ...")
    df = pd.read_parquet(INPUT)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Sanity check: CO_SH6 should be populated for most rows
    sh6_null_pct = 100 * df["CO_SH6"].isna().sum() / len(df)
    print(f"  CO_SH6 null rate: {sh6_null_pct:.1f}%")
    if sh6_null_pct > 5:
        print("  WARNING: >5% of rows have no CO_SH6 — check NCM reference join in 02_enrich.py")

    print("\nBuilding marts ...")
    build_all(df)

    print(f"\nAll marts written to {OUT}/")
    print("Stage 3 complete.\n")
    print("Verification checklist:")
    print("  [ ] mart_chapter_month.parquet — Chapter 30 (pharma) appears in top-10 by VL_FOB")
    print("  [ ] mart_ncm4_country.parquet  — CO_PAIS '160' (China) rows present")
    print("  [ ] mart_sh6_via.parquet       — MEDIAN_FREIGHT_PCT_FOB populated for air (CO_VIA=4)")
    print("  [ ] No mart has more rows than enriched.parquet (no cartesian explosion)")


if __name__ == "__main__":
    main()
