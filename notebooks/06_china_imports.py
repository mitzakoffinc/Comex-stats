# %% [markdown]
# # 06 — China Import Intelligence
#
# **Purpose:** Deep-dive analysis of Brazilian imports from China. Surfaces high-value HS6 segments,
# key trade routes, and freight patterns to identify prospects for QEntrega and Itatibense Transportes.
# **Input:** `outputs/data/enriched.parquet`
# **Filter:** `CO_PAIS = "160"` (China)
# **Primary metric:** FOB USD (`VL_FOB`)
# **HS-level:** HS6 (`CO_SH6`) — broader and internationally comparable vs. NCM8.
#
# Run `pipeline/00_ingest.py → 01_clean.py → 02_enrich.py → 03_marts.py` before executing.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "outputs" / "data"
CHART_DIR    = PROJECT_ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (12, 4)})

CHINA_CODE = "160"

print("Loading enriched data ...")
enr = pd.read_parquet(DATA_DIR / "enriched.parquet")
print(f"  Total rows: {len(enr):,}")

cn = enr[enr["CO_PAIS"] == CHINA_CODE].copy()
print(f"  China rows: {len(cn):,}  ({100*len(cn)/len(enr):.1f}% of total)")

if len(cn) == 0:
    raise RuntimeError("No China rows found. Check CO_PAIS codes in PAIS.csv reference.")

# Subset with valid HS6 — used for all HS6-level analysis
cn_hs6 = cn[cn["CO_SH6"].notna()].copy()
sh6_coverage = 100 * len(cn_hs6) / len(cn)
print(f"  China rows with CO_SH6: {len(cn_hs6):,}  ({sh6_coverage:.1f}% of China rows)")
if sh6_coverage < 95:
    print("  WARNING: <95% CO_SH6 coverage — HS6-level analysis has gaps. Check NCM reference join.")

# %% [markdown]
# ## 1. Volume & Value Overview
# China's total contribution to Brazilian imports in this dataset (2025–2026).

# %%
annual_all = (
    enr.groupby("CO_ANO")[["KG_LIQUIDO", "VL_FOB"]]
    .sum()
    .rename(columns={"KG_LIQUIDO": "KG_ALL", "VL_FOB": "FOB_ALL"})
)
annual_cn = (
    cn.groupby("CO_ANO")[["KG_LIQUIDO", "VL_FOB"]]
    .sum()
    .rename(columns={"KG_LIQUIDO": "KG_CN", "VL_FOB": "FOB_CN"})
)
annual = annual_all.join(annual_cn, how="left")
annual["CN_SHARE_FOB_PCT"] = (annual["FOB_CN"] / annual["FOB_ALL"] * 100).round(1)
annual["CN_SHARE_KG_PCT"]  = (annual["KG_CN"]  / annual["KG_ALL"]  * 100).round(1)

display_df = pd.DataFrame({
    "Total FOB (USD bn)":    (annual["FOB_ALL"] / 1e9).round(2),
    "China FOB (USD bn)":    (annual["FOB_CN"]  / 1e9).round(2),
    "China FOB share (%)": annual["CN_SHARE_FOB_PCT"],
    "China net kg (M MT)":   (annual["KG_CN"]   / 1e9).round(2),
    "China kg share (%)": annual["CN_SHARE_KG_PCT"],
})
print(display_df.to_string())
print("\nNote: 2026 is a partial year (~Q1 only). Totals are not YoY-comparable without annualizing.")

# %%
years  = annual.index.astype(str).tolist()
x      = np.arange(len(years))
w      = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(x - w/2, annual["FOB_ALL"] / 1e9, w, label="All imports", color="#1f77b4", alpha=0.85)
axes[0].bar(x + w/2, annual["FOB_CN"]  / 1e9, w, label="China",       color="#d62728", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(years)
axes[0].set_ylabel("FOB USD (billions)")
axes[0].set_title("FOB Value: China vs. Total Imports")
axes[0].legend()

axes[1].bar(years, annual["CN_SHARE_FOB_PCT"], color="#d62728", alpha=0.85)
axes[1].set_ylabel("China share (%)")
axes[1].set_title("China Share of Total Import FOB (%)")
for i, v in enumerate(annual["CN_SHARE_FOB_PCT"]):
    axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(CHART_DIR / "china_annual_overview.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Top HS6 Segments — by FOB Value and Volume
# Aggregated at the 6-digit HS level. Descriptions come from the corresponding NCM8 entry (first match per HS6 group).
# Unit prices exclude IQR-flagged outliers; volume totals include all rows.

# %%
# Base aggregation at HS6 level
hs6_base = (
    cn_hs6.groupby("CO_SH6")
    .agg(
        NO_NCM_POR  = ("NO_NCM_POR",  "first"),
        KG_LIQUIDO  = ("KG_LIQUIDO",  "sum"),
        VL_FOB      = ("VL_FOB",      "sum"),
        N_OPS       = ("VL_FOB",      "count"),
    )
    .reset_index()
)

# Unit price and freight stats per HS6 (outlier-excluded for price stats)
price_rows = []
for hs6, grp in cn_hs6.groupby("CO_SH6"):
    clean_price = grp.loc[~grp["is_outlier_unitval"], "UNIT_FOB_PER_KG"].dropna()
    fret_pct    = grp["FREIGHT_PCT_FOB"].dropna()
    price_rows.append({
        "CO_SH6":             hs6,
        "MEDIAN_UNIT_FOB":    round(float(clean_price.median()), 2) if len(clean_price) else np.nan,
        "P10_UNIT_FOB":       round(float(clean_price.quantile(0.10)), 2) if len(clean_price) > 5 else np.nan,
        "P90_UNIT_FOB":       round(float(clean_price.quantile(0.90)), 2) if len(clean_price) > 5 else np.nan,
        "MEDIAN_FREIGHT_PCT": round(float(fret_pct.median() * 100), 2) if len(fret_pct) else np.nan,
    })

price_df = pd.DataFrame(price_rows)
hs6_full = hs6_base.merge(price_df, on="CO_SH6", how="left")
hs6_full["VL_FOB_mn"] = (hs6_full["VL_FOB"] / 1e6).round(1)
hs6_full["KG_MT"]     = (hs6_full["KG_LIQUIDO"] / 1e3).round(0).astype("Int64")

print(f"Total HS6 codes from China: {len(hs6_full):,}")

print("\nTop 25 HS6 segments — ranked by FOB USD:")
top_fob = (
    hs6_full.nlargest(25, "VL_FOB")
    [["CO_SH6", "NO_NCM_POR", "VL_FOB_mn", "KG_MT", "N_OPS", "MEDIAN_UNIT_FOB", "MEDIAN_FREIGHT_PCT"]]
    .reset_index(drop=True)
)
top_fob.columns = ["HS6", "Description (NCM8 label)", "FOB USD (M)", "Net kg (MT)",
                   "N Ops", "Median USD/kg", "Median freight %"]
print(top_fob.to_string())

# %%
top15 = hs6_full.nlargest(15, "VL_FOB").sort_values("VL_FOB")
labels = [
    f"{row.CO_SH6} — {str(row.NO_NCM_POR or '')[:45]}"
    for _, row in top15.iterrows()
]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(labels, top15["VL_FOB_mn"], color="#d62728", alpha=0.85)
ax.set_xlabel("FOB USD (millions)")
ax.set_title("Top 15 HS6 Segments — China Imports by FOB Value (2025–2026)")

for bar in bars:
    ax.text(
        bar.get_width() + bar.get_width() * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"${bar.get_width():.0f}M",
        va="center", fontsize=8,
    )

plt.tight_layout()
plt.savefig(CHART_DIR / "china_top_hs6_fob.png", bbox_inches="tight")
plt.show()

print("\nTop 20 HS6 segments — ranked by net weight (kg):")
top_kg = (
    hs6_full.nlargest(20, "KG_LIQUIDO")
    [["CO_SH6", "NO_NCM_POR", "KG_MT", "VL_FOB_mn", "MEDIAN_UNIT_FOB"]]
    .reset_index(drop=True)
)
top_kg.columns = ["HS6", "Description (NCM8 label)", "Net kg (MT)", "FOB USD (M)", "Median USD/kg"]
print(top_kg.to_string())

# %% [markdown]
# ## 3. Port of Entry (URF)
# Which Brazilian customs posts (URFs) receive Chinese cargo? Concentration expected at Santos (sea), GRU/VCP (air), and Itajaí.

# %%
urf_cn = (
    cn.groupby(["CO_URF", "NO_URF"])
    .agg(KG_LIQUIDO=("KG_LIQUIDO", "sum"), VL_FOB=("VL_FOB", "sum"), N_OPS=("VL_FOB", "count"))
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
)
urf_cn["VL_FOB_mn"]  = (urf_cn["VL_FOB"] / 1e6).round(1)
urf_cn["FOB_share"]  = (urf_cn["VL_FOB"] / urf_cn["VL_FOB"].sum() * 100).round(1)
urf_cn["cumul_share"] = urf_cn["FOB_share"].cumsum().round(1)

print("Top 15 ports of entry — China imports, ranked by FOB:")
print(
    urf_cn[["CO_URF", "NO_URF", "VL_FOB_mn", "FOB_share", "cumul_share", "N_OPS"]]
    .head(15)
    .reset_index(drop=True)
    .rename(columns={
        "CO_URF": "URF Code", "NO_URF": "Port / Airport",
        "VL_FOB_mn": "FOB USD (M)", "FOB_share": "Share (%)",
        "cumul_share": "Cumul. share (%)", "N_OPS": "N Ops",
    })
    .to_string()
)
top5_share = urf_cn.head(5)["FOB_share"].sum()
print(f"\nTop-5 URFs account for {top5_share:.0f}% of China import FOB.")

# %%
top_urf = urf_cn.head(12)
port_labels = top_urf["NO_URF"].fillna(top_urf["CO_URF"]).str[:35]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(port_labels, top_urf["VL_FOB_mn"], color="#1f77b4", alpha=0.85)
ax.set_ylabel("FOB USD (millions)")
ax.set_title("Top 12 Ports of Entry — China Imports by FOB Value")
plt.xticks(rotation=38, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "china_urf_fob.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Destination State (SG_UF_NCM)
# Which Brazilian states import from China? SP/MG concentration expected (manufacturing hubs).

# %%
uf_cn = (
    cn.groupby("SG_UF_NCM")
    .agg(KG_LIQUIDO=("KG_LIQUIDO", "sum"), VL_FOB=("VL_FOB", "sum"), N_OPS=("VL_FOB", "count"))
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
)
uf_cn["VL_FOB_mn"] = (uf_cn["VL_FOB"] / 1e6).round(1)
uf_cn["FOB_share"] = (uf_cn["VL_FOB"] / uf_cn["VL_FOB"].sum() * 100).round(1)

print(
    uf_cn[["SG_UF_NCM", "VL_FOB_mn", "FOB_share", "N_OPS"]]
    .head(15)
    .reset_index(drop=True)
    .rename(columns={"SG_UF_NCM": "State", "VL_FOB_mn": "FOB USD (M)",
                     "FOB_share": "Share (%)", "N_OPS": "N Ops"})
    .to_string()
)
top3 = uf_cn.head(3)
print(f"Top-3 states: {', '.join(top3['SG_UF_NCM'].tolist())} — "
      f"{top3['FOB_share'].sum():.0f}% of China imports by FOB.")

# %%
top_uf = uf_cn.head(12).sort_values("VL_FOB")

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(top_uf["SG_UF_NCM"], top_uf["VL_FOB_mn"], color="#2ca02c", alpha=0.85)
ax.set_xlabel("FOB USD (millions)")
ax.set_title("Top 12 Destination States — China Imports by FOB Value")
plt.tight_layout()
plt.savefig(CHART_DIR / "china_uf_fob.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Transport Mode Mix (CO_VIA)
# Air vs. sea vs. road split. Air freight dominates FOB share (high-value goods); sea dominates by weight.

# %%
via_cn = (
    cn.groupby(["CO_VIA", "NO_VIA"])
    .agg(
        KG_LIQUIDO  = ("KG_LIQUIDO", "sum"),
        VL_FOB      = ("VL_FOB",     "sum"),
        N_OPS       = ("VL_FOB",     "count"),
    )
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
)
via_cn["VL_FOB_mn"] = (via_cn["VL_FOB"] / 1e6).round(1)
via_cn["FOB_share"] = (via_cn["VL_FOB"] / via_cn["VL_FOB"].sum() * 100).round(1)
via_cn["KG_share"]  = (via_cn["KG_LIQUIDO"] / via_cn["KG_LIQUIDO"].sum() * 100).round(1)

print(
    via_cn[["CO_VIA", "NO_VIA", "VL_FOB_mn", "FOB_share", "KG_share", "N_OPS"]]
    .reset_index(drop=True)
    .rename(columns={"CO_VIA": "Mode Code", "NO_VIA": "Mode",
                     "VL_FOB_mn": "FOB USD (M)", "FOB_share": "FOB share (%)",
                     "KG_share": "KG share (%)", "N_OPS": "N Ops"})
    .to_string()
)

# %%
mode_labels = via_cn["NO_VIA"].fillna(via_cn["CO_VIA"]).str[:20].tolist()
x = np.arange(len(mode_labels))
w = 0.35

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x - w/2, via_cn["FOB_share"],  w, label="FOB share (%)",     color="#d62728", alpha=0.85)
ax.bar(x + w/2, via_cn["KG_share"],   w, label="Net-kg share (%)",  color="#1f77b4", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(mode_labels, rotation=30, ha="right")
ax.set_ylabel("Share (%)")
ax.set_title("Transport Mode Split — China Imports by FOB Value vs. Net Weight")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "china_mode_split.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Monthly Trend 2025 → 2026
# Volume and FOB value over time. Use only for seasonality/direction signals — 2026 is a partial year (~Q1).

# %%
monthly = (
    cn.groupby(["CO_ANO", "CO_MES"])
    .agg(KG_LIQUIDO=("KG_LIQUIDO", "sum"), VL_FOB=("VL_FOB", "sum"), N_OPS=("VL_FOB", "count"))
    .reset_index()
    .sort_values(["CO_ANO", "CO_MES"])
)
monthly["PERIODO"] = (
    monthly["CO_ANO"].astype(str) + "-" + monthly["CO_MES"].astype(str).str.zfill(2)
)
monthly["VL_FOB_mn"] = (monthly["VL_FOB"] / 1e6).round(1)
monthly["KG_kMT"]    = (monthly["KG_LIQUIDO"] / 1e6).round(1)

x      = list(range(len(monthly)))
labels = monthly["PERIODO"].tolist()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(x, monthly["VL_FOB_mn"], marker="o", color="#d62728", linewidth=2)
axes[0].fill_between(x, monthly["VL_FOB_mn"], alpha=0.15, color="#d62728")
axes[0].set_ylabel("FOB USD (millions)")
axes[0].set_title("China Import Monthly Trend — FOB Value")
axes[0].grid(axis="y", alpha=0.4)

first_2026 = monthly.index[monthly["CO_ANO"] == 2026]
if len(first_2026):
    axes[0].axvspan(x[first_2026[0]] - 0.5, x[-1] + 0.5, alpha=0.07, color="gray", label="2026 partial")
    axes[0].legend(fontsize=8)

axes[1].bar(x, monthly["KG_kMT"], color="#1f77b4", alpha=0.85)
axes[1].set_ylabel("Net weight (thousand MT)")
axes[1].set_title("China Import Monthly Trend — Net Weight")
axes[1].grid(axis="y", alpha=0.4)
if len(first_2026):
    axes[1].axvspan(x[first_2026[0]] - 0.5, x[-1] + 0.5, alpha=0.07, color="gray")

plt.xticks(x, labels, rotation=45, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "china_monthly_trend.png", bbox_inches="tight")
plt.show()
print("Shaded area = 2026 (partial year, ~Q1 only). Do not annualize without explicit adjustment.")

# %% [markdown]
# ## 7. Unit Value by HS6 — Pricing Intelligence
# Median FOB USD/kg by HS6 segment (outlier-flagged rows excluded from price stats, included in volume totals).
# High unit value = premium/specialty goods; also correlates with air freight preference.

# %%
# High unit value segments (min 50 operations for reliability)
reliable = hs6_full[hs6_full["N_OPS"] >= 50]

print("Top 25 HS6 — Highest median unit value (min 50 ops, outlier-excluded):")
high_val = (
    reliable[reliable["MEDIAN_UNIT_FOB"].notna()]
    .nlargest(25, "MEDIAN_UNIT_FOB")
    [["CO_SH6", "NO_NCM_POR", "MEDIAN_UNIT_FOB", "P10_UNIT_FOB", "P90_UNIT_FOB",
      "MEDIAN_FREIGHT_PCT", "VL_FOB_mn", "N_OPS"]]
    .reset_index(drop=True)
)
high_val.columns = ["HS6", "Description", "Median USD/kg", "P10 USD/kg", "P90 USD/kg",
                    "Median freight %", "FOB USD (M)", "N Ops"]
print(high_val.to_string())

print("\nTop 25 HS6 — Lowest median unit value (bulk/commodity, min 50 ops):")
low_val = (
    reliable[reliable["MEDIAN_UNIT_FOB"].notna()]
    .nsmallest(25, "MEDIAN_UNIT_FOB")
    [["CO_SH6", "NO_NCM_POR", "MEDIAN_UNIT_FOB", "MEDIAN_FREIGHT_PCT", "VL_FOB_mn", "N_OPS"]]
    .reset_index(drop=True)
)
low_val.columns = ["HS6", "Description", "Median USD/kg", "Median freight %", "FOB USD (M)", "N Ops"]
print(low_val.to_string())

# %%
scatter_df = hs6_full[
    hs6_full["MEDIAN_UNIT_FOB"].notna() &
    hs6_full["MEDIAN_FREIGHT_PCT"].notna() &
    (hs6_full["N_OPS"] >= 50)
].copy()

if len(scatter_df) < 5:
    print("Insufficient data for scatter (need FREIGHT_PCT_FOB populated; check VL_FRETE coverage in Section 5).")
else:
    scatter_df["bubble_size"] = (
        np.sqrt(scatter_df["VL_FOB"] / scatter_df["VL_FOB"].max()) * 400
    ).clip(lower=20)

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        scatter_df["MEDIAN_UNIT_FOB"],
        scatter_df["MEDIAN_FREIGHT_PCT"],
        s=scatter_df["bubble_size"],
        c=np.log1p(scatter_df["VL_FOB"]),
        cmap="YlOrRd",
        alpha=0.6,
        edgecolors="gray",
        linewidths=0.4,
    )
    plt.colorbar(sc, label="log(Total FOB USD)")
    ax.set_xlabel("Median unit value (FOB USD / net kg)")
    ax.set_ylabel("Median freight cost as % of FOB")
    ax.set_title("China HS6 Segments: Unit Value vs. Freight Intensity\n"
                 "(bubble size = total FOB; colour = log FOB)")

    for _, row in scatter_df.nlargest(10, "VL_FOB").iterrows():
        ax.annotate(
            row["CO_SH6"],
            (row["MEDIAN_UNIT_FOB"], row["MEDIAN_FREIGHT_PCT"]),
            fontsize=7, ha="center", va="bottom",
        )

    plt.tight_layout()
    plt.savefig(CHART_DIR / "china_unit_vs_freight.png", bbox_inches="tight")
    plt.show()
    print("Note: MEDIAN_FREIGHT_PCT = median(VL_FRETE / VL_FOB) × 100.")
    print("Dimensionless — not affected by net-weight vs. chargeable-weight estimation.")
    print(f"Segments plotted: {len(scatter_df):,} (≥50 ops + freight data available)")

# %% [markdown]
# ## 8. "So What?" — Opportunity Segments for QEntrega & Itatibense
# Cross-reference HS6 × URF (port) to surface the highest-volume, highest-frequency China trade lanes.
# These are the product–port combinations where logistics volume justifies targeted sales investment.

# %%
# HS6 × URF × State cross — China only
hs6_urf = (
    cn_hs6.groupby(["CO_SH6", "NO_NCM_POR", "CO_URF", "NO_URF", "SG_UF_NCM"])
    .agg(
        VL_FOB     = ("VL_FOB",      "sum"),
        N_OPS      = ("VL_FOB",      "count"),
        KG_LIQUIDO = ("KG_LIQUIDO",  "sum"),
    )
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
)
hs6_urf["VL_FOB_mn"] = (hs6_urf["VL_FOB"] / 1e6).round(1)

print(f"Unique HS6 × URF × State combinations (China imports): {len(hs6_urf):,}")
print("\nTop 20 — by FOB value:")
top_lanes = (
    hs6_urf.head(20)
    [["CO_SH6", "NO_NCM_POR", "NO_URF", "SG_UF_NCM", "VL_FOB_mn", "N_OPS"]]
    .reset_index(drop=True)
)
top_lanes.columns = ["HS6", "Product (NCM label)", "Port", "Dest. state", "FOB USD (M)", "N Ops"]
print(top_lanes.to_string())

# %%
N_OPS_MIN = 200
FOB_MIN   = 500_000   # USD

priority = hs6_urf[
    (hs6_urf["N_OPS"] >= N_OPS_MIN) &
    (hs6_urf["VL_FOB"] >= FOB_MIN)
].sort_values("N_OPS", ascending=False)

print(f"Priority lanes (≥{N_OPS_MIN} ops AND ≥${FOB_MIN/1e3:.0f}K FOB):")
print(f"  {len(priority):,} HS6 × URF × State combinations identified.")

if len(priority) == 0:
    print(f"  No lanes met thresholds. Lower N_OPS_MIN or FOB_MIN above.")
else:
    print(
        priority[["CO_SH6", "NO_NCM_POR", "NO_URF", "SG_UF_NCM", "VL_FOB_mn", "N_OPS"]]
        .head(30)
        .reset_index(drop=True)
        .rename(columns={
            "CO_SH6":     "HS6",
            "NO_NCM_POR": "Product (NCM label)",
            "NO_URF":     "Port",
            "SG_UF_NCM":  "Dest. state",
            "VL_FOB_mn":  "FOB USD (M)",
            "N_OPS":      "N Ops",
        })
        .to_string()
    )

print("""
=== READING GUIDE FOR QEntrega / Itatibense ===

High N Ops + high FOB  →  recurring, valuable cargo — prime target for freight contracts
Air-dominant modes     →  urgency cargo, high-margin (check Section 5 for mode detail)
                           QEntrega priority: GRU/VCP air lanes
Sea/road URFs          →  Itatibense inland drayage opportunity (Santos → SP/MG interior)
SP and MG destinations →  Both operators' service areas — match against current customer list

PROSPECTING NEXT STEP:
Cross-reference these HS6 × URF × state combinations with your current customer list.
Gaps = segments where Chinese goods move in volume at a port you serve,
       but no current customer in that HS6 sector.

DATA LIMITATION:
Public Comex Stat is anonymized — no importer CNPJs or company names.
To unlock named importers in these segments, add Logcomex or ImportGenius data
(DI-level extract) in a future phase.
""")
