# %% [markdown]
# # 01 — Data Profile & Quality Sign-off
#
# **Purpose:** Validate pipeline output quality before any analysis begins.
# **Input:** `outputs/data/raw_combined.parquet`, `outputs/data/clean.parquet`, `outputs/data/enriched.parquet`
# **Output:** Go/no-go sign-off with documented data quality baseline.
#
# Run `pipeline/00_ingest.py → 01_clean.py → 02_enrich.py` before executing this notebook.

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def _find_project_root() -> Path:
    start = Path(globals().get("__file__", Path.cwd() / "x")).resolve().parent
    for p in [start, *start.parents]:
        if (p / "Config" / "config.xlsx").exists():
            return p
    raise FileNotFoundError("Project root not found (no Config/config.xlsx upward of cwd)")


PROJECT_ROOT = _find_project_root()
DATA_DIR     = PROJECT_ROOT / "outputs" / "data"
CHART_DIR    = PROJECT_ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (12, 4)})

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

_con        = duckdb.connect()
_raw_path   = str(DATA_DIR / "raw_combined.parquet").replace("\\", "/")
_clean_path = str(DATA_DIR / "clean.parquet").replace("\\", "/")
_enr_path   = str(DATA_DIR / "enriched.parquet").replace("\\", "/")

# %% [markdown]
# ## 1. Raw data — row counts and date coverage

# %%
_raw_all_cols = [d[0] for d in _con.execute(f"SELECT * FROM read_parquet('{_raw_path}') LIMIT 0").description]
raw = _con.execute(f"""
    SELECT CO_ANO, CO_MES, CO_NCM, CO_PAIS, SG_UF_NCM, CO_VIA, CO_URF,
           KG_LIQUIDO, VL_FOB, VL_FRETE, VL_SEGURO
    FROM read_parquet('{_raw_path}')
""").df()
print(f"Total rows (raw): {len(raw):,}")
print(f"All schema columns ({len(_raw_all_cols)}): {_raw_all_cols}")
print(f"Loaded columns   : {list(raw.columns)}")
print(f"Memory: {raw.memory_usage(deep=True).sum()/1e6:.1f} MB")

# %%
# Rows per year × month
ym = raw.groupby(["CO_ANO", "CO_MES"]).size().reset_index(name="N_ROWS")
print("Row counts by year × month:")
pivot = ym.pivot(index="CO_MES", columns="CO_ANO", values="N_ROWS").fillna(0).astype(int)
display(pivot)

# %%
fig, ax = plt.subplots()
for year, grp in ym.groupby("CO_ANO"):
    ax.plot(grp["CO_MES"], grp["N_ROWS"] / 1000, marker="o", label=str(int(year)))
ax.set_xlabel("Month")
ax.set_ylabel("Rows (thousands)")
ax.set_title("Row count per month")
ax.legend(title="Year")
plt.tight_layout()
plt.savefig(CHART_DIR / "profile_rows_per_month.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Null rates — critical fields

# %%
critical_cols = ["CO_NCM", "CO_PAIS", "SG_UF_NCM", "CO_VIA", "CO_URF",
                 "KG_LIQUIDO", "VL_FOB", "VL_FRETE", "VL_SEGURO"]

null_summary = []
for col in critical_cols:
    if col in raw.columns:
        n = raw[col].isna().sum()
        null_summary.append({"column": col, "null_count": n,
                              "null_pct": 100 * n / len(raw)})

null_df = pd.DataFrame(null_summary).set_index("column")
null_df["null_pct"] = null_df["null_pct"].round(2)
display(null_df)

# Flag high-null columns
high_null = null_df[null_df["null_pct"] > 5]
if len(high_null):
    print(f"\nWARNING: columns with >5% nulls: {high_null.index.tolist()}")
else:
    print("\nOK: no critical columns exceed 5% null rate.")

# %%
# VL_FRETE null rate is expected to be high — document it
frete_null_pct = 100 * raw["VL_FRETE"].isna().sum() / len(raw)
print(f"VL_FRETE null rate: {frete_null_pct:.1f}%")
print("Note: Missing VL_FRETE is common in Comex Stat public data.")
print("Freight analysis will be scoped to rows where VL_FRETE is populated.")

# %% [markdown]
# ## 3. Anomaly check — negative and zero values

# %%
anomalies = {
    "Negative VL_FOB":      int((raw["VL_FOB"] < 0).sum()),
    "Negative KG_LIQUIDO":  int((raw["KG_LIQUIDO"] < 0).sum()),
    "Negative VL_FRETE":    int((raw["VL_FRETE"] < 0).sum()),
    "Zero VL_FOB":          int((raw["VL_FOB"] == 0).sum()),
    "Zero KG_LIQUIDO":      int((raw["KG_LIQUIDO"] == 0).sum()),
}

anom_df = pd.Series(anomalies).rename("count").to_frame()
anom_df["pct"] = (anom_df["count"] / len(raw) * 100).round(2)
display(anom_df)

if any(v > 0 for v in [anomalies["Negative VL_FOB"],
                        anomalies["Negative KG_LIQUIDO"]]):
    print("\nWARNING: negative monetary/weight values found — review before using in totals.")
else:
    print("\nOK: no negative FOB or weight values.")

# %% [markdown]
# ## 4. NCM validation — clean stage

# %%
_clean_all_cols = [d[0] for d in _con.execute(f"SELECT * FROM read_parquet('{_clean_path}') LIMIT 0").description]
clean = _con.execute(f"""
    SELECT CO_ANO, CO_NCM, is_outlier_unitval, is_zero_weight, is_complete_period
    FROM read_parquet('{_clean_path}')
""").df()
invalid_path = DATA_DIR / "invalid_ncm.parquet"

n_clean   = len(clean)
n_raw     = len(raw)
n_invalid = n_raw - n_clean

if invalid_path.exists():
    _inv_str = str(invalid_path).replace("\\", "/")
    invalid  = _con.execute(f"SELECT CO_NCM FROM read_parquet('{_inv_str}')").df()
    n_invalid = len(invalid)
    print(f"Invalid NCM rows quarantined: {n_invalid:,} ({100*n_invalid/n_raw:.2f}%)")
    if n_invalid > 0:
        print(f"Sample bad NCMs: {invalid['CO_NCM'].unique()[:10].tolist()}")
else:
    print("No invalid_ncm.parquet found — all NCMs passed validation.")

print(f"\nClean rows retained: {n_clean:,} ({100*n_clean/n_raw:.2f}%)")
print(f"Clean schema columns ({len(_clean_all_cols)}): {_clean_all_cols}")

# %% [markdown]
# ## 5. Outlier and quality flags

# %%
flag_cols = ["is_outlier_unitval", "is_zero_weight", "is_complete_period"]
for col in flag_cols:
    if col in clean.columns:
        n = clean[col].sum()
        print(f"  {col:30s}: {n:,} ({100*n/len(clean):.2f}%)")

# %%
# Outlier unit values by year — are they concentrated in specific periods?
if "is_outlier_unitval" in clean.columns:
    outlier_by_year = (
        clean.groupby("CO_ANO")["is_outlier_unitval"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "outliers", "count": "total"})
    )
    outlier_by_year["pct"] = (outlier_by_year["outliers"] / outlier_by_year["total"] * 100).round(2)
    display(outlier_by_year)

# %% [markdown]
# ## 6. Enriched data — reference join quality

# %%
_enr_all_cols = [d[0] for d in _con.execute(f"SELECT * FROM read_parquet('{_enr_path}') LIMIT 0").description]
enriched = _con.execute(f"""
    SELECT CO_ANO, CO_NCM, CO_SH6, NO_NCM_POR,
           CO_PAIS, NO_PAIS_ING, NO_URF, NO_VIA,
           KG_LIQUIDO, VL_FOB, VL_FRETE,
           CO_CAPITULO
    FROM read_parquet('{_enr_path}')
""").df()
print(f"Enriched rows: {len(enriched):,}")
print(f"All schema columns ({len(_enr_all_cols)}): {_enr_all_cols}")

# %%
# Reference join quality
join_checks = [
    ("NO_NCM_POR",  "NCM description (PT)"),
    ("CO_SH6",      "HS6 code"),
    ("NO_PAIS_ING", "Country name (EN)"),
    ("NO_URF",      "Port/URF name"),
    ("NO_VIA",      "Transport mode name"),
]

join_df = []
for col, label in join_checks:
    if col in enriched.columns:
        n_null = enriched[col].isna().sum()
        join_df.append({"field": col, "description": label,
                        "unmatched": n_null,
                        "unmatched_pct": round(100 * n_null / len(enriched), 2)})

join_report = pd.DataFrame(join_df).set_index("field")
display(join_report)

high_unmatch = join_report[join_report["unmatched_pct"] > 1]
if len(high_unmatch):
    print(f"\nWARNING: join fields >1% unmatched: {high_unmatch.index.tolist()}")
else:
    print("\nOK: all reference joins have <1% unmatched rows.")

# %%
# CO_SH6 null rate check (required for China analysis at HS6 level)
sh6_null = enriched["CO_SH6"].isna().sum()
sh6_null_pct = 100 * sh6_null / len(enriched)
print(f"CO_SH6 null rate: {sh6_null:,} rows ({sh6_null_pct:.2f}%)")
if sh6_null_pct > 5:
    print("WARNING: >5% of rows missing HS6 — China analysis at HS6 level will have gaps.")
else:
    print("OK: CO_SH6 coverage is sufficient for HS6-level China analysis.")

# %% [markdown]
# ## 7. Encoding check — mojibake detection in text fields

# %%
# Mojibake patterns: replacement char or garbled sequences common in mis-decoded Latin-1
MOJIBAKE_PATTERN = r'[�\x80-\x9f]|Ã[£©ção]|Ã\xa7|â€'

text_fields = ["NO_NCM_POR", "NO_PAIS_ING", "NO_URF", "NO_VIA"]
for col in text_fields:
    if col not in enriched.columns:
        continue
    sample = enriched[col].dropna()
    bad = sample[sample.str.contains(MOJIBAKE_PATTERN, regex=True, na=False)]
    if len(bad):
        print(f"WARNING: {col}: {len(bad):,} possible mojibake rows")
        print(f"   Sample: {bad.unique()[:5].tolist()}")
    else:
        print(f"OK: {col}: no encoding artifacts detected")

# %%
# Spot-check Portuguese NCM descriptions (should contain accented characters, not garbage)
if "NO_NCM_POR" in enriched.columns:
    sample_ncm = enriched[["CO_NCM", "NO_NCM_POR"]].dropna().drop_duplicates("CO_NCM").head(10)
    display(sample_ncm.reset_index(drop=True))

# %% [markdown]
# ## 8. Market totals sanity check

# %%
# Total FOB by year — reasonable order of magnitude check
totals = (
    enriched.groupby("CO_ANO")[["KG_LIQUIDO", "VL_FOB"]]
    .sum()
    .assign(
        VL_FOB_bn = lambda d: d["VL_FOB"] / 1e9,
        KG_MM_mt  = lambda d: d["KG_LIQUIDO"] / 1e9,
    )
)
print("Totals by year (VL_FOB in USD billions, KG in million metric tons):")
display(totals[["VL_FOB_bn", "KG_MM_mt"]].round(2))

months_per_year = raw.groupby("CO_ANO")["CO_MES"].nunique()
for year, n_months in months_per_year.items():
    if n_months < 12:
        print(f"\nNote: {year} is a partial year ({n_months} months). "
              "Do not compare raw totals for YoY.")

# %%
# Top-5 chapters by FOB (sanity: pharma/chem should be in top-10)
chapter_totals = (
    enriched.groupby("CO_CAPITULO")["VL_FOB"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
chapter_totals["VL_FOB_bn"] = (chapter_totals["VL_FOB"] / 1e9).round(2)
print("Top-10 chapters by FOB USD:")
display(chapter_totals[["CO_CAPITULO", "VL_FOB_bn"]])

if "30" in chapter_totals["CO_CAPITULO"].values:
    print("\nOK: Chapter 30 (pharma) appears in top-10 — data looks plausible.")
else:
    print("\nWARNING: Chapter 30 (pharma) NOT in top-10 — verify the data.")

# %%
# Confirm China rows are present
china_rows = enriched[enriched["CO_PAIS"] == "160"]
print(f"China (CO_PAIS='160') rows: {len(china_rows):,}")
print(f"China share of total FOB: {100 * china_rows['VL_FOB'].sum() / enriched['VL_FOB'].sum():.1f}%")

if len(china_rows) == 0:
    print("\nWARNING: NO China rows found — check CO_PAIS codes in PAIS.csv.")
else:
    print("\nOK: China rows present — China notebook can proceed.")

# %% [markdown]
# ## 9. Freight data coverage

# %%
# VL_FRETE coverage by transport mode
if "NO_VIA" in enriched.columns:
    frete_by_via = (
        enriched.groupby("NO_VIA")
        .apply(lambda g: pd.Series({
            "total_rows": len(g),
            "frete_populated": g["VL_FRETE"].notna().sum(),
            "frete_pct": round(100 * g["VL_FRETE"].notna().sum() / len(g), 1),
        }), include_groups=False)
        .sort_values("total_rows", ascending=False)
    )
    print("VL_FRETE population rate by transport mode:")
    display(frete_by_via)

# %% [markdown]
# ## 10. Go / No-go summary

# %%
checks = {}

# 1. Row count reasonable
checks["Row count > 1M"] = len(enriched) > 1_000_000

# 2. NCM loss < 1%
ncm_loss = 1 - len(clean) / len(raw)
checks["NCM validation loss < 1%"] = ncm_loss < 0.01

# 3. CO_SH6 coverage > 95%
checks["CO_SH6 coverage > 95%"] = sh6_null_pct < 5

# 4. China rows present
checks["China rows present"] = len(china_rows) > 0

# 5. Pharma chapter in top-10
checks["Chapter 30 in top-10 FOB"] = "30" in chapter_totals["CO_CAPITULO"].values

# 6. No negative FOB
checks["No negative VL_FOB"] = int((raw["VL_FOB"] < 0).sum()) == 0

print("=" * 50)
print("  DATA QUALITY SIGN-OFF")
print("=" * 50)
all_pass = True
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("  ALL CHECKS PASSED — proceed to analysis notebooks.")
else:
    print("  SOME CHECKS FAILED — review issues above before proceeding.")
