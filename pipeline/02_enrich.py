"""
Stage 2 — Enrich

Input:  outputs/data/clean.parquet
Output: outputs/data/enriched.parquet

What it does:
  - Left-joins all reference tables (NCM, PAIS, URF, VIA, UF)
  - Adds derived fields (VL_CIF, UNIT_FOB_PER_KG already set; adds FREIGHT_*, PERIODO, etc.)
  - Reports unmatched join rates so data gaps are visible

Run from project root:
    python pipeline/02_enrich.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFS_DIR     = PROJECT_ROOT / "Data" / "References"
INPUT        = PROJECT_ROOT / "outputs" / "data" / "clean.parquet"
OUTPUT       = PROJECT_ROOT / "outputs" / "data" / "enriched.parquet"


# ---------------------------------------------------------------------------
# Reference loaders
# ---------------------------------------------------------------------------

def _read_ref(filename: str, string_cols: list[str], **kwargs) -> pd.DataFrame:
    """Read a reference CSV with encoding cascade."""
    path = REFS_DIR / filename
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                dtype={c: str for c in string_cols},
                quotechar='"',
                low_memory=False,
                **kwargs,
            )
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip().str.strip('"')
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def load_ncm() -> pd.DataFrame:
    df = _read_ref("NCM.csv", string_cols=["CO_NCM", "CO_SH6", "CO_PPE", "CO_FAT_AGREG"])
    return df[["CO_NCM", "CO_SH6", "NO_NCM_POR", "NO_NCM_ING", "CO_PPE", "CO_FAT_AGREG"]].drop_duplicates("CO_NCM")


def load_pais() -> pd.DataFrame:
    df = _read_ref("PAIS.csv", string_cols=["CO_PAIS", "CO_PAIS_ISON3", "CO_PAIS_ISOA3"])
    return df[["CO_PAIS", "CO_PAIS_ISON3", "CO_PAIS_ISOA3", "NO_PAIS", "NO_PAIS_ING"]].drop_duplicates("CO_PAIS")


def load_urf() -> pd.DataFrame:
    df = _read_ref("URF.csv", string_cols=["CO_URF"])
    return df[["CO_URF", "NO_URF"]].drop_duplicates("CO_URF")


def load_via() -> pd.DataFrame:
    df = _read_ref("VIA.csv", string_cols=["CO_VIA"])
    return df[["CO_VIA", "NO_VIA"]].drop_duplicates("CO_VIA")


def load_uf() -> pd.DataFrame:
    df = _read_ref("UF.csv", string_cols=["CO_UF", "SG_UF"])
    # UF reference may use SG_UF as the join key matching SG_UF_NCM in main data
    cols = [c for c in df.columns if c in ("SG_UF", "CO_UF", "NO_UF", "NO_REGIAO")]
    return df[cols].drop_duplicates(subset=[c for c in ("SG_UF", "CO_UF") if c in df.columns])


# ---------------------------------------------------------------------------
# Join & report
# ---------------------------------------------------------------------------

def join_and_report(df: pd.DataFrame, ref: pd.DataFrame, left_on: str, ref_name: str) -> pd.DataFrame:
    before = len(df)
    merged = df.merge(ref, left_on=left_on, right_on=ref.columns[0], how="left")
    unmatched = merged[ref.columns[1]].isna().sum()
    pct = 100 * unmatched / max(before, 1)
    print(f"  {ref_name:20s}: {unmatched:,} unmatched ({pct:.1f}%)")
    return merged


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    # CIF (secondary — FOB is primary)
    df["VL_CIF"] = df["VL_FOB"] + df["VL_FRETE"].fillna(0) + df["VL_SEGURO"].fillna(0)

    # Unit prices — UNIT_FOB_PER_KG already set in clean stage; recompute for safety
    mask_kg = df["KG_LIQUIDO"] > 0
    df["UNIT_FOB_PER_KG"] = np.where(mask_kg, df["VL_FOB"] / df["KG_LIQUIDO"], np.nan)
    df["UNIT_CIF_PER_KG"] = np.where(mask_kg, df["VL_CIF"] / df["KG_LIQUIDO"], np.nan)

    # Freight proxies
    # NOTE: KG_LIQUIDO is net weight. Actual chargeable weight = max(gross, CBM/6000).
    # FREIGHT_PER_KG is a proxy only; FREIGHT_PCT_FOB is dimensionless and more reliable.
    df["FREIGHT_PER_KG"] = np.where(
        mask_kg & df["VL_FRETE"].notna(),
        df["VL_FRETE"] / df["KG_LIQUIDO"],
        np.nan,
    )
    df["FREIGHT_PCT_FOB"] = np.where(
        df["VL_FOB"] > 0,
        df["VL_FRETE"].fillna(0) / df["VL_FOB"],
        np.nan,
    )

    # Hierarchy codes
    df["CO_CAPITULO"] = df["CO_NCM"].str[:2]   # HS chapter
    df["CO_POSICAO"]  = df["CO_NCM"].str[:4]   # HS heading (already added in clean, but idempotent)

    # Period integer YYYYMM for easy sorting and filtering
    df["PERIODO"] = (
        df["CO_ANO"].astype("Int64") * 100 + df["CO_MES"].astype("Int64")
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  STAGE 2 — ENRICH")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run pipeline/01_clean.py first.")

    print(f"\nReading {INPUT.name} ...")
    df = pd.read_parquet(INPUT)
    print(f"  Loaded {len(df):,} rows")

    # --- Load references ---
    print("\nLoading reference tables ...")
    ncm_ref  = load_ncm()
    pais_ref = load_pais()
    urf_ref  = load_urf()
    via_ref  = load_via()
    uf_ref   = load_uf()

    # --- Joins ---
    print("\nJoin match rates (unmatched = no lookup entry found):")
    df = join_and_report(df, ncm_ref,  left_on="CO_NCM",    ref_name="NCM")
    df = join_and_report(df, pais_ref, left_on="CO_PAIS",   ref_name="PAIS")
    df = join_and_report(df, urf_ref,  left_on="CO_URF",    ref_name="URF")
    df = join_and_report(df, via_ref,  left_on="CO_VIA",    ref_name="VIA")

    # UF join: match SG_UF_NCM in main data to SG_UF in reference
    uf_key = "SG_UF" if "SG_UF" in uf_ref.columns else uf_ref.columns[0]
    df = join_and_report(df, uf_ref,   left_on="SG_UF_NCM", ref_name="UF")

    # --- Derived fields ---
    print("\nAdding derived fields ...")
    df = add_derived_fields(df)

    print(f"\n  Final columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  Rows: {len(df):,}")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # --- Write ---
    print(f"\nWriting {OUTPUT} ...")
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        OUTPUT,
        compression="snappy",
    )
    print(f"  Done. {OUTPUT.stat().st_size/1e6:.1f} MB")
    print("\nStage 2 complete. Next: pipeline/03_marts.py")


if __name__ == "__main__":
    main()
