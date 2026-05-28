"""
Stage 1 — Clean

Input:  outputs/data/raw_combined.parquet
Output: outputs/data/clean.parquet
        outputs/data/invalid_ncm.parquet

What it does:
  - Validates NCM format (8 digits); quarantines bad rows
  - Detects unit-value outliers (IQR per NCM-4 group); flags, never drops
  - Flags zero-weight rows separately (unit-based goods)
  - Marks period completeness (complete vs partial year-months)

Run from project root:
    python pipeline/01_clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "outputs" / "data" / "raw_combined.parquet"
OUT_CLEAN = PROJECT_ROOT / "outputs" / "data" / "clean.parquet"
OUT_INVALID_NCM = PROJECT_ROOT / "outputs" / "data" / "invalid_ncm.parquet"

# 2026 is a partial year — mark complete periods only through 2025-12
# Update this set if 2026 data is refreshed with more months.
KNOWN_COMPLETE_PERIODS = {
    (2025, m) for m in range(1, 13)
}


def validate_ncm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into valid and invalid NCM rows. NCM must be exactly 8 digits."""
    digits_only = df["CO_NCM"].str.replace(r"\D", "", regex=True)
    valid_mask = digits_only.str.len() == 8
    return df[valid_mask].copy(), df[~valid_mask].copy()


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute UNIT_FOB_PER_KG and flag IQR outliers per NCM-4 group.

    Rows where KG_LIQUIDO == 0 or VL_FOB == 0 get UNIT_FOB_PER_KG = NaN
    and is_outlier_unitval = False (flagged separately by is_zero_weight).

    IQR fence: Q1 - 3*IQR, Q3 + 3*IQR  (wide fence — only catches extreme
    data-entry errors like grams vs. kg or single unit vs. full batch).
    """
    # Unit price — null when weight is zero to avoid division by zero
    df["UNIT_FOB_PER_KG"] = np.where(
        df["KG_LIQUIDO"] > 0,
        df["VL_FOB"] / df["KG_LIQUIDO"],
        np.nan,
    )

    df["CO_POSICAO"] = df["CO_NCM"].str[:4]  # NCM-4 for grouping

    df["is_outlier_unitval"] = False

    # Only flag rows that have a unit price
    has_price = df["UNIT_FOB_PER_KG"].notna() & (df["VL_FOB"] > 0)
    group_col = "CO_POSICAO"

    def iqr_bounds(series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        return (series < lo) | (series > hi)

    # Vectorised group-level outlier flag
    outlier_mask = (
        df[has_price]
        .groupby(group_col)["UNIT_FOB_PER_KG"]
        .transform(iqr_bounds)
        .reindex(df.index, fill_value=False)
        .astype(bool)
    )
    df["is_outlier_unitval"] = outlier_mask

    return df


def flag_zero_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where KG_LIQUIDO == 0 but VL_FOB > 0 (unit-based goods)."""
    df["is_zero_weight"] = (df["KG_LIQUIDO"] == 0) | (df["KG_LIQUIDO"].isna())
    return df


def flag_period_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each row as belonging to a complete or partial period."""
    df["is_complete_period"] = df.apply(
        lambda r: (int(r["CO_ANO"]), int(r["CO_MES"])) in KNOWN_COMPLETE_PERIODS
        if pd.notna(r["CO_ANO"]) and pd.notna(r["CO_MES"])
        else False,
        axis=1,
    )
    return df


def print_summary(valid: pd.DataFrame, invalid: pd.DataFrame) -> None:
    total = len(valid) + len(invalid)
    print(f"\n  Total input rows   : {total:,}")
    print(f"  Valid NCM rows     : {len(valid):,} ({100*len(valid)/total:.2f}%)")
    print(f"  Invalid NCM rows   : {len(invalid):,} ({100*len(invalid)/total:.2f}%)")
    if len(invalid) > 0:
        print(f"  Sample bad NCMs    : {invalid['CO_NCM'].unique()[:10].tolist()}")

    out_count = valid["is_outlier_unitval"].sum()
    zero_w    = valid["is_zero_weight"].sum()
    incomplete = (~valid["is_complete_period"]).sum()

    print(f"\n  Outlier unit-value flags  : {out_count:,} ({100*out_count/len(valid):.2f}%)")
    print(f"  Zero-weight flags         : {zero_w:,} ({100*zero_w/len(valid):.2f}%)")
    print(f"  Incomplete-period rows    : {incomplete:,} ({100*incomplete/len(valid):.2f}%)")


def main() -> None:
    print("=" * 60)
    print("  STAGE 1 — CLEAN")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run pipeline/00_ingest.py first."
        )

    print(f"\nReading {INPUT.name} ...")
    df = pd.read_parquet(INPUT)
    print(f"  Loaded {len(df):,} rows")

    # --- NCM validation ---
    print("\nValidating NCM format ...")
    valid, invalid = validate_ncm(df)

    # --- Derived flags ---
    print("Flagging zero-weight rows ...")
    valid = flag_zero_weight(valid)

    print("Flagging outlier unit values (IQR per NCM-4, wide fence 3×IQR) ...")
    valid = flag_outliers(valid)

    print("Flagging period completeness ...")
    valid = flag_period_completeness(valid)

    # --- Summary ---
    print_summary(valid, invalid)

    # --- Write outputs ---
    print(f"\nWriting {OUT_CLEAN} ...")
    pq.write_table(
        pa.Table.from_pandas(valid, preserve_index=False),
        OUT_CLEAN,
        compression="snappy",
    )
    print(f"  {len(valid):,} rows → {OUT_CLEAN.stat().st_size/1e6:.1f} MB")

    if len(invalid) > 0:
        print(f"Writing {OUT_INVALID_NCM} ...")
        pq.write_table(
            pa.Table.from_pandas(invalid, preserve_index=False),
            OUT_INVALID_NCM,
            compression="snappy",
        )
        print(f"  {len(invalid):,} rows quarantined")
    else:
        print("  No invalid NCM rows — invalid_ncm.parquet not created")

    print("\nStage 1 complete. Next: pipeline/02_enrich.py")


if __name__ == "__main__":
    main()
