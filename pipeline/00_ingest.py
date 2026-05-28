"""
Stage 0 — Ingest & Validate

Reads both import CSVs (2025, 2026), applies encoding cascade, enforces dtypes,
emits a data profile report, and writes outputs/data/raw_combined.parquet.

Run from the project root:
    python pipeline/00_ingest.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "Imports Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    2025: DATA_DIR / "IMP_2025.csv",
    2026: DATA_DIR / "IMP_2026.csv",
}

OUTPUT_PARQUET = OUTPUT_DIR / "raw_combined.parquet"

# ---------------------------------------------------------------------------
# Dtype spec — string cols must stay string to preserve leading zeros
# ---------------------------------------------------------------------------
STRING_COLS = ["CO_NCM", "CO_UNID", "CO_PAIS", "SG_UF_NCM", "CO_VIA", "CO_URF"]
NUMERIC_INT_COLS = ["CO_ANO", "CO_MES"]
NUMERIC_FLOAT_COLS = ["QT_ESTAT", "KG_LIQUIDO", "VL_FOB", "VL_FRETE", "VL_SEGURO"]


def _detect_encoding(path: Path) -> str:
    """Try encodings in cascade order; return the first that reads the header."""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.readline()
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No supported encoding could read {path}")


def read_csv(path: Path, year: int) -> pd.DataFrame:
    """Read one import CSV with defensive settings."""
    enc = _detect_encoding(path)
    print(f"  [{year}] encoding: {enc}")

    df = pd.read_csv(
        path,
        sep=";",
        encoding=enc,
        dtype={c: str for c in STRING_COLS},
        quotechar='"',
        decimal=".",       # Comex Stat exports use period decimal
        low_memory=False,
    )

    # Strip any residual quote characters from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip('"')

    # Cast numeric columns explicitly
    for col in NUMERIC_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in NUMERIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Source year tag (redundant with CO_ANO but useful for filtering)
    df["SOURCE_YEAR"] = year

    return df


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def null_rate(series: pd.Series) -> str:
    n = series.isna().sum()
    pct = 100 * n / max(len(series), 1)
    return f"{n:,} ({pct:.1f}%)"


def profile(df: pd.DataFrame, year: int) -> dict:
    rows = len(df)
    ano_vals = df["CO_ANO"].dropna().unique().tolist()
    mes_min = int(df["CO_MES"].min()) if df["CO_MES"].notna().any() else None
    mes_max = int(df["CO_MES"].max()) if df["CO_MES"].notna().any() else None

    neg_fob = int((df["VL_FOB"] < 0).sum())
    neg_kg  = int((df["KG_LIQUIDO"] < 0).sum())
    neg_frt = int((df["VL_FRETE"] < 0).sum())

    zero_fob = int((df["VL_FOB"] == 0).sum())
    zero_kg  = int((df["KG_LIQUIDO"] == 0).sum())

    # Strict duplicate: full row equality across the business key
    biz_key = ["CO_ANO", "CO_MES", "CO_NCM", "CO_PAIS", "SG_UF_NCM", "CO_VIA", "CO_URF"]
    dup_key_count = df.duplicated(subset=biz_key).sum()

    # NCM format check: should be exactly 8 digits
    ncm_bad = df["CO_NCM"].str.replace(r"\D", "", regex=True).str.len().ne(8).sum()

    return {
        "year": year,
        "rows": rows,
        "anos": ano_vals,
        "mes_range": f"{mes_min}–{mes_max}",
        "unique_ncm": df["CO_NCM"].nunique(),
        "unique_pais": df["CO_PAIS"].nunique(),
        "unique_urf": df["CO_URF"].nunique(),
        "unique_uf": df["SG_UF_NCM"].nunique(),
        "null_ncm": null_rate(df["CO_NCM"]),
        "null_pais": null_rate(df["CO_PAIS"]),
        "null_urf": null_rate(df["CO_URF"]),
        "null_kg": null_rate(df["KG_LIQUIDO"]),
        "null_fob": null_rate(df["VL_FOB"]),
        "null_frete": null_rate(df["VL_FRETE"]),
        "null_seguro": null_rate(df["VL_SEGURO"]),
        "neg_fob": neg_fob,
        "neg_kg": neg_kg,
        "neg_frete": neg_frt,
        "zero_fob": zero_fob,
        "zero_kg": zero_kg,
        "dup_biz_key": int(dup_key_count),
        "ncm_bad_format": int(ncm_bad),
    }


def print_profile(p: dict) -> None:
    yr = p["year"]
    print(f"\n{'='*60}")
    print(f"  FILE: IMP_{yr}.csv")
    print(f"{'='*60}")
    print(f"  Rows              : {p['rows']:,}")
    print(f"  CO_ANO values     : {p['anos']}")
    print(f"  CO_MES range      : {p['mes_range']}")
    if yr == 2026:
        print(f"  *** PARTIAL YEAR: 2026 data through month {p['mes_range'].split('–')[-1]} only.")
        print(f"      Do NOT use raw 2026 totals for YoY comparisons without annualizing.")
    print(f"  Unique NCMs       : {p['unique_ncm']:,}")
    print(f"  Unique countries  : {p['unique_pais']:,}")
    print(f"  Unique URFs       : {p['unique_urf']:,}")
    print(f"  Unique UF states  : {p['unique_uf']:,}")
    print()
    print(f"  Null rates:")
    print(f"    CO_NCM     : {p['null_ncm']}")
    print(f"    CO_PAIS    : {p['null_pais']}")
    print(f"    CO_URF     : {p['null_urf']}")
    print(f"    KG_LIQUIDO : {p['null_kg']}")
    print(f"    VL_FOB     : {p['null_fob']}")
    print(f"    VL_FRETE   : {p['null_frete']}")
    print(f"    VL_SEGURO  : {p['null_seguro']}")
    print()
    print(f"  Anomalies:")
    print(f"    Negative VL_FOB     : {p['neg_fob']:,}")
    print(f"    Negative KG_LIQUIDO : {p['neg_kg']:,}")
    print(f"    Negative VL_FRETE   : {p['neg_frete']:,}")
    print(f"    Zero VL_FOB         : {p['zero_fob']:,}")
    print(f"    Zero KG_LIQUIDO     : {p['zero_kg']:,}")
    print(f"    Dup biz-key rows    : {p['dup_biz_key']:,}")
    print(f"    Bad NCM format      : {p['ncm_bad_format']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  STAGE 0 — INGEST & VALIDATE")
    print("=" * 60)

    frames = []
    profiles = []

    for year, path in FILES.items():
        if not path.exists():
            print(f"\n[ERROR] File not found: {path}")
            sys.exit(1)

        print(f"\nReading {path.name} ...")
        df = read_csv(path, year)
        p = profile(df, year)
        profiles.append(p)
        print_profile(p)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\n{'='*60}")
    print("  COMBINED DATASET")
    print(f"{'='*60}")
    print(f"  Total rows        : {len(combined):,}")
    print(f"  Columns           : {list(combined.columns)}")
    print(f"  Memory usage      : {combined.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # --- Write parquet ---
    print(f"\nWriting {OUTPUT_PARQUET} ...")
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, OUTPUT_PARQUET, compression="snappy")
    size_mb = OUTPUT_PARQUET.stat().st_size / 1e6
    print(f"  Done. File size: {size_mb:.1f} MB")
    print("\nStage 0 complete. Next: pipeline/01_clean.py")


if __name__ == "__main__":
    main()
