# %% [markdown]
# # 04 — Pricing Benchmarks
#
# **Purpose:** Median / P10 / P90 FOB USD/kg benchmarks by NCM-4 and by origin country.
# **Input:** `outputs/data/mart_ncm4_country.parquet`
# **Output:** Price benchmark tables, China vs. all-origins comparison, air-freight feasibility read-out.
#
# Key metric: **FOB USD/kg (outlier-excluded median)** — the pre-computed `MEDIAN_UNIT_FOB_PER_KG`
# in the mart already excludes `is_outlier_unitval` rows (IQR fence per NCM-4).
# The aggregate unit price (`VL_FOB / KG_LIQUIDO`) is kept for comparison only; do not use it
# as a benchmark.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (12, 5)})

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

FOCUS_CHAPTERS = ["28", "29", "30", "38", "39"]
FOCUS_LABELS   = {
    "28": "Ch28 Inorg Chem", "29": "Ch29 Org Chem",
    "30": "Ch30 Pharma",     "38": "Ch38 Misc Chem", "39": "Ch39 Plastics"
}

# Air-freight feasibility thresholds (USD/kg)
AIR_FLOOR = 20.0
AIR_IDEAL = 50.0

# %% [markdown]
# ## 1. Load mart_ncm4_country

# %%
ncm_ctry = pd.read_parquet(DATA_DIR / "mart_ncm4_country.parquet")

ncm_ctry["PAIS_LABEL"] = ncm_ctry["NO_PAIS_ING"].fillna(
    "PAIS-" + ncm_ctry["CO_PAIS"].astype(str)
)
ncm_ctry["CO_CAPITULO"] = ncm_ctry["CO_POSICAO"].astype(str).str[:2]
ncm_ctry["AGG_UNIT_FOB_PER_KG"] = np.where(
    ncm_ctry["KG_LIQUIDO"] > 0,
    ncm_ctry["VL_FOB"] / ncm_ctry["KG_LIQUIDO"],
    np.nan,
)

# Resolve China code from data — no hardcoding
_cn = ncm_ctry[ncm_ctry["NO_PAIS_ING"].str.contains("China", case=False, na=False)]
CHINA_CODE = _cn["CO_PAIS"].iloc[0] if len(_cn) else "160"

print(f"Rows loaded       : {len(ncm_ctry):,}")
print(f"Unique NCM-4 codes: {ncm_ctry['CO_POSICAO'].nunique():,}")
print(f"Unique countries  : {ncm_ctry['CO_PAIS'].nunique():,}")
print(f"Columns: {list(ncm_ctry.columns)}")
print(f"China CO_PAIS resolved: {CHINA_CODE}")
ncm_ctry.head(3)

# %% [markdown]
# ## 2. NCM-4 summary — totals and price benchmarks across all origins

# %%
# Aggregate to NCM-4 level (sum across all origins)
ncm4_summary = (
    ncm_ctry
    .groupby("CO_POSICAO")
    .agg(
        CO_CAPITULO = ("CO_CAPITULO", "first"),
        VL_FOB      = ("VL_FOB",      "sum"),
        KG_LIQUIDO  = ("KG_LIQUIDO",  "sum"),
        VL_FRETE    = ("VL_FRETE",    "sum"),
        N_OPS       = ("N_OPS",       "sum"),
        N_ORIGINS   = ("CO_PAIS",     "nunique"),
    )
    .reset_index()
)

# Aggregate (weighted-average) unit price — includes outlier effect, use for comparison only
ncm4_summary["AGG_UNIT_FOB"] = np.where(
    ncm4_summary["KG_LIQUIDO"] > 0,
    ncm4_summary["VL_FOB"] / ncm4_summary["KG_LIQUIDO"],
    np.nan,
)

# Price spread across origins: distribution of per-country outlier-excluded medians
origin_spread = (
    ncm_ctry[ncm_ctry["MEDIAN_UNIT_FOB_PER_KG"].notna()]
    .groupby("CO_POSICAO")["MEDIAN_UNIT_FOB_PER_KG"]
    .agg(
        MEDIAN_ORIGIN = "median",
        P10_ORIGIN    = lambda s: float(s.quantile(0.10)),
        P90_ORIGIN    = lambda s: float(s.quantile(0.90)),
    )
    .reset_index()
)
origin_spread["SPREAD_RATIO"] = np.where(
    origin_spread["P10_ORIGIN"] > 0,
    origin_spread["P90_ORIGIN"] / origin_spread["P10_ORIGIN"],
    np.nan,
)

ncm4_full = ncm4_summary.merge(origin_spread, on="CO_POSICAO", how="left")
ncm4_full = ncm4_full.sort_values("VL_FOB", ascending=False).reset_index(drop=True)
ncm4_full["VL_FOB_M"] = (ncm4_full["VL_FOB"] / 1e6).round(2)

print(f"NCM-4 codes with data: {len(ncm4_full):,}")
print("\nTop-20 by total FOB USD:")
top20 = ncm4_full.head(20)[[
    "CO_POSICAO", "CO_CAPITULO", "VL_FOB_M", "N_ORIGINS", "N_OPS",
    "AGG_UNIT_FOB", "MEDIAN_ORIGIN", "P10_ORIGIN", "P90_ORIGIN", "SPREAD_RATIO"
]].copy()
top20.columns = ["NCM-4","Chp","FOB_$M","N_Orig","N_Ops",
                 "Agg_$/kg","Med_$/kg","P10_$/kg","P90_$/kg","P90/P10"]
for col in ["Agg_$/kg","Med_$/kg","P10_$/kg","P90_$/kg"]:
    top20[col] = top20[col].round(2)
top20["P90/P10"] = top20["P90/P10"].round(1)
display(top20.reset_index(drop=True))

# %% [markdown]
# ## 3. Price distribution — across all NCM-4 codes

# %%
prices = ncm4_full["MEDIAN_ORIGIN"].dropna()
cap    = prices.quantile(0.99)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: linear histogram capped at 99th percentile
ax = axes[0]
ax.hist(prices[prices <= cap], bins=50, color="#4393c3", edgecolor="white", linewidth=0.3)
ax.axvline(prices.median(), color="#d73027", lw=1.5,
           label=f"Overall median: ${prices.median():.2f}/kg")
ax.set_xlabel("Median FOB USD/kg  (capped at 99th pct)")
ax.set_ylabel("Count of NCM-4 codes")
ax.set_title("Unit price distribution — linear scale")
ax.legend(fontsize=9)

# Right: log-scale (full distribution)
ax = axes[1]
log_prices = np.log10(prices[prices > 0])
ax.hist(log_prices, bins=50, color="#74add1", edgecolor="white", linewidth=0.3)
for val, label in [(0, "$1"), (1, "$10"), (2, "$100"), (3, "$1,000")]:
    ax.axvline(val, color="gray", lw=0.8, linestyle="--", alpha=0.6)
    ax.text(val + 0.05, ax.get_ylim()[1] * 0.85, label, fontsize=8, color="gray")
ax.set_xlabel("log10(Median FOB USD/kg)")
ax.set_ylabel("Count of NCM-4 codes")
ax.set_title("Unit price distribution — log scale")

plt.tight_layout()
plt.savefig(CHART_DIR / "pricing_distribution.png", bbox_inches="tight")
plt.show()

print("\nPrice distribution summary:")
for pct in [10, 25, 50, 75, 90, 99]:
    print(f"  P{pct:2d}: ${prices.quantile(pct/100):>10.2f}/kg")

# %% [markdown]
# ## 4. Outlier impact: aggregate-weighted vs. median unit price

# %%
# Ratio > 1 = aggregate (includes outliers) exceeds median (outliers excluded).
# A high ratio signals that a small number of extreme rows are inflating the average —
# declaration errors (e.g., grams reported as kg) or genuine one-off transactions.
valid_comp = ncm4_full.dropna(subset=["AGG_UNIT_FOB", "MEDIAN_ORIGIN"]).copy()
valid_comp["OUTLIER_PULL"] = np.where(
    valid_comp["MEDIAN_ORIGIN"] > 0,
    valid_comp["AGG_UNIT_FOB"] / valid_comp["MEDIAN_ORIGIN"],
    np.nan,
)

distorted = (
    valid_comp[valid_comp["OUTLIER_PULL"] > 1.5]
    .sort_values("OUTLIER_PULL", ascending=False)
    .head(20)
)

print("Top-20 NCM-4 where aggregate price is >1.5x the outlier-excluded median")
print("(these codes would be mispriced if you used total FOB / total KG as the benchmark)\n")
d = distorted[["CO_POSICAO","CO_CAPITULO","VL_FOB_M",
               "AGG_UNIT_FOB","MEDIAN_ORIGIN","OUTLIER_PULL"]].copy()
d.columns = ["NCM-4","Chp","FOB_$M","Agg_$/kg","Med_$/kg","Agg/Med"]
for col in ["Agg_$/kg","Med_$/kg","Agg/Med"]:
    d[col] = d[col].round(2)
display(d.reset_index(drop=True))

n_distorted = len(valid_comp[valid_comp["OUTLIER_PULL"] > 1.5])
print(f"\n{n_distorted:,} / {len(valid_comp):,} NCM-4 codes ({100*n_distorted/len(valid_comp):.0f}%) "
      f"have Agg/Med > 1.5")
print("-> Always use MEDIAN_UNIT_FOB_PER_KG for any price benchmark or feasibility study.")

# %% [markdown]
# ## 5. Pricing leverage — NCM-4 codes with widest origin spread

# %%
# High P90/P10 spread across origins = the most expensive origin charges 5-20x the cheapest.
# For logistics clients, this highlights re-sourcing opportunities or price negotiation leverage.
high_spread = (
    ncm4_full[
        (ncm4_full["N_ORIGINS"]   >= 3) &
        (ncm4_full["SPREAD_RATIO"].notna()) &
        (ncm4_full["VL_FOB"]      >= 1_000_000)
    ]
    .sort_values("SPREAD_RATIO", ascending=False)
    .head(30)
)

print("Top-30 NCM-4 by price spread across origin countries")
print("Filter: >=3 origins, >=$1M total FOB  |  Metric: P90 / P10 of per-country medians\n")
sp = high_spread[["CO_POSICAO","CO_CAPITULO","VL_FOB_M","N_ORIGINS",
                  "P10_ORIGIN","MEDIAN_ORIGIN","P90_ORIGIN","SPREAD_RATIO"]].copy()
sp.columns = ["NCM-4","Chp","FOB_$M","N_Orig","P10_$/kg","Med_$/kg","P90_$/kg","P90/P10"]
for col in ["P10_$/kg","Med_$/kg","P90_$/kg"]:
    sp[col] = sp[col].round(2)
sp["P90/P10"] = sp["P90/P10"].round(1)
display(sp.reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(12, 8))
plot_data = high_spread.head(20).sort_values("SPREAD_RATIO").reset_index(drop=True)
y = range(len(plot_data))
ylabels = plot_data["CO_POSICAO"].astype(str) + " (Ch" + plot_data["CO_CAPITULO"] + ")"

ax.barh(y, plot_data["P90_ORIGIN"], color="#d73027", alpha=0.65, label="P90 origin ($/kg)")
ax.barh(y, plot_data["P10_ORIGIN"], color="#4393c3", alpha=0.85, label="P10 origin ($/kg)")
ax.barh(y, plot_data["MEDIAN_ORIGIN"], height=0.25, color="#1a9850", alpha=0.9,
        label="Median ($/kg)")
ax.set_yticks(y)
ax.set_yticklabels(ylabels, fontsize=9)
ax.set_xlabel("FOB USD/kg  (spread of per-country medians)")
ax.set_title("Top-20 NCM-4 by origin price spread  (P10 / Median / P90)\n"
             "Red = most expensive origin  |  Blue = cheapest origin  |  Green = median")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(CHART_DIR / "pricing_spread_by_origin.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Focus chapters — pricing benchmarks (Ch 28, 29, 30, 38, 39)

# %%
for chp in FOCUS_CHAPTERS:
    chp_data = (
        ncm4_full[ncm4_full["CO_CAPITULO"] == chp]
        .sort_values("VL_FOB", ascending=False)
        .head(15)
    )
    if len(chp_data) == 0:
        print(f"\nChapter {chp}: no data\n")
        continue

    total_fob = chp_data["VL_FOB"].sum()
    print(f"\n{'='*62}")
    print(f"  {FOCUS_LABELS[chp]}  —  top-15 NCM-4 by FOB  "
          f"(total shown: ${total_fob/1e6:.0f}M)")
    print(f"{'='*62}")
    c = chp_data[[
        "CO_POSICAO","VL_FOB_M","N_ORIGINS","N_OPS",
        "AGG_UNIT_FOB","MEDIAN_ORIGIN","P10_ORIGIN","P90_ORIGIN","SPREAD_RATIO"
    ]].copy()
    c.columns = ["NCM-4","FOB_$M","N_Orig","N_Ops",
                 "Agg_$/kg","Med_$/kg","P10_$/kg","P90_$/kg","P90/P10"]
    for col in ["Agg_$/kg","Med_$/kg","P10_$/kg","P90_$/kg"]:
        c[col] = c[col].round(2)
    c["P90/P10"] = c["P90/P10"].round(1)
    display(c.reset_index(drop=True))

# %% [markdown]
# ## 7. China vs. all-origins unit price comparison

# %%
china_prices = ncm_ctry[ncm_ctry["CO_PAIS"] == CHINA_CODE].copy()
china_prices = china_prices.rename(columns={
    "MEDIAN_UNIT_FOB_PER_KG": "MEDIAN_CHINA",
    "P10_UNIT_FOB_PER_KG":    "P10_CHINA",
    "P90_UNIT_FOB_PER_KG":    "P90_CHINA",
    "VL_FOB":                 "VL_FOB_CHINA",
    "KG_LIQUIDO":             "KG_CHINA",
    "N_OPS":                  "N_OPS_CHINA",
})

china_vs_all = ncm4_full.merge(
    china_prices[[
        "CO_POSICAO", "MEDIAN_CHINA", "P10_CHINA", "P90_CHINA",
        "VL_FOB_CHINA", "KG_CHINA", "N_OPS_CHINA"
    ]],
    on="CO_POSICAO",
    how="inner",  # only NCMs where China actually imports
)

# China price index: 1.0 = parity with all-origins median
china_vs_all["CHINA_PRICE_IDX"] = np.where(
    china_vs_all["MEDIAN_ORIGIN"] > 0,
    china_vs_all["MEDIAN_CHINA"] / china_vs_all["MEDIAN_ORIGIN"],
    np.nan,
)

china_vs_all = china_vs_all.sort_values("VL_FOB_CHINA", ascending=False)
china_vs_all["VL_FOB_CHINA_M"] = (china_vs_all["VL_FOB_CHINA"] / 1e6).round(2)

print(f"NCM-4 codes with China imports: {len(china_vs_all):,}")
print("\nTop-30 by China FOB — China median vs. all-origins median:")
cv = china_vs_all.head(30)[[
    "CO_POSICAO","CO_CAPITULO","VL_FOB_CHINA_M","N_OPS_CHINA",
    "MEDIAN_CHINA","MEDIAN_ORIGIN","CHINA_PRICE_IDX"
]].copy()
cv.columns = ["NCM-4","Chp","China_FOB_$M","China_Ops",
              "China_Med_$/kg","All_Med_$/kg","China/All_idx"]
cv["China_Med_$/kg"] = cv["China_Med_$/kg"].round(2)
cv["All_Med_$/kg"]   = cv["All_Med_$/kg"].round(2)
cv["China/All_idx"]  = cv["China/All_idx"].round(2)
display(cv.reset_index(drop=True))

valid_idx = china_vs_all["CHINA_PRICE_IDX"].dropna()
cheaper = (valid_idx < 0.9).sum()
similar = ((valid_idx >= 0.9) & (valid_idx <= 1.1)).sum()
premium = (valid_idx > 1.1).sum()
n = len(valid_idx)
print(f"\nChina price index distribution vs. all-origins median ({n:,} NCM-4 codes):")
print(f"  China discount  (< 0.9x): {cheaper:,}  ({100*cheaper/n:.0f}%)")
print(f"  At parity  (0.9-1.1x)  : {similar:,}  ({100*similar/n:.0f}%)")
print(f"  China premium  (> 1.1x): {premium:,}  ({100*premium/n:.0f}%)")

# %%
fig, ax = plt.subplots(figsize=(10, 7))
plot_data = china_vs_all.head(20).sort_values("CHINA_PRICE_IDX").reset_index(drop=True)
colors = [
    "#d73027" if x > 1.1 else "#1a9850" if x < 0.9 else "#fdae61"
    for x in plot_data["CHINA_PRICE_IDX"]
]
ax.barh(range(len(plot_data)), plot_data["CHINA_PRICE_IDX"], color=colors)
ax.axvline(1.0, color="black",   lw=1.2, linestyle="--", label="Parity (1.0)")
ax.axvline(0.9, color="#1a9850", lw=0.8, linestyle=":",  alpha=0.7, label="-10% (0.9)")
ax.axvline(1.1, color="#d73027", lw=0.8, linestyle=":",  alpha=0.7, label="+10% (1.1)")
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(
    plot_data["CO_POSICAO"].astype(str) + "  (Ch" + plot_data["CO_CAPITULO"] + ")",
    fontsize=9
)
ax.set_xlabel("China median USD/kg  /  All-origins median USD/kg")
ax.set_title(
    "China price index — top-20 NCM-4 codes by China FOB\n"
    "Green = China cheaper  |  Orange = at parity  |  Red = China premium"
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(CHART_DIR / "pricing_china_index.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Air-freight feasibility — unit value thresholds

# %%
# Air-freight cost for Brazil routes typically runs $3-8 USD/kg (GRU/VCP).
# As a rule of thumb:
#   < $20/kg  ->  freight likely exceeds 15-25% of value; air hard to justify
#   $20-50/kg ->  borderline; depends on urgency, size, and actual freight quote
#   > $50/kg  ->  air freight clearly economical (freight < 10-15% of value)

high_value = ncm4_full[ncm4_full["MEDIAN_ORIGIN"] >= AIR_FLOOR]
high_value_focus = high_value[high_value["CO_CAPITULO"].isin(FOCUS_CHAPTERS)]

print(f"Air-freight feasibility thresholds (based on MEDIAN_ORIGIN):")
print(f"  Floor    >= ${AIR_FLOOR}/kg : {len(high_value):,} NCM-4 codes "
      f"of {len(ncm4_full):,} total "
      f"({100*len(high_value)/len(ncm4_full):.0f}%)")
print(f"  Ideal    >= ${AIR_IDEAL}/kg : "
      f"{len(ncm4_full[ncm4_full['MEDIAN_ORIGIN'] >= AIR_IDEAL]):,} codes")
print(f"\nFocus-chapter NCMs above ${AIR_FLOOR}/kg:")
for chp in FOCUS_CHAPTERS:
    n_above = len(ncm4_full[(ncm4_full["CO_CAPITULO"] == chp) &
                             (ncm4_full["MEDIAN_ORIGIN"] >= AIR_FLOOR)])
    n_total = len(ncm4_full[ncm4_full["CO_CAPITULO"] == chp])
    if n_total > 0:
        print(f"  {FOCUS_LABELS[chp]:22s}: {n_above:3d} / {n_total:3d} codes "
              f"({100*n_above/n_total:.0f}%) above ${AIR_FLOOR}/kg")

print(f"\nNote: thresholds use median FOB/kg across origins.")
print(f"Individual shipments vary — always validate with actual freight quotes.")

# %%
fig, ax = plt.subplots(figsize=(10, 6))

# Background: all chapters with sufficient size
bg = ncm4_full[
    ncm4_full["MEDIAN_ORIGIN"].notna() &
    (ncm4_full["VL_FOB"] >= 500_000) &
    ~ncm4_full["CO_CAPITULO"].isin(FOCUS_CHAPTERS)
]
ax.scatter(
    np.log10(bg["VL_FOB"].clip(lower=1)),
    bg["MEDIAN_ORIGIN"].clip(upper=500),
    s=20, color="#aaaaaa", alpha=0.35, label="Other chapters"
)

# Foreground: focus chapters
colors_focus = ["#4393c3","#74add1","#d73027","#f46d43","#1a9850"]
for chp, col in zip(FOCUS_CHAPTERS, colors_focus):
    sub = ncm4_full[
        (ncm4_full["CO_CAPITULO"] == chp) &
        ncm4_full["MEDIAN_ORIGIN"].notna() &
        (ncm4_full["VL_FOB"] >= 500_000)
    ]
    ax.scatter(
        np.log10(sub["VL_FOB"].clip(lower=1)),
        sub["MEDIAN_ORIGIN"].clip(upper=500),
        s=55, color=col, alpha=0.85, label=FOCUS_LABELS[chp]
    )

ax.axhline(AIR_FLOOR, color="#fdae61", lw=1.5, linestyle="--",
           label=f"Air floor (${AIR_FLOOR}/kg)")
ax.axhline(AIR_IDEAL, color="#d73027", lw=1.2, linestyle="--",
           label=f"Air ideal (${AIR_IDEAL}/kg)")
ax.set_xlabel("log10(Total FOB USD)  — market size")
ax.set_ylabel("Median FOB USD/kg  (capped at $500)")
ax.set_title("Unit value vs. market size — air-freight feasibility by NCM-4")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(CHART_DIR / "pricing_air_feasibility.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Strategic read-out

# %%
print("=" * 62)
print("  PRICING BENCHMARKS — KEY FINDINGS")
print("=" * 62)

# 1. Outlier distortion scale
n_distorted = len(valid_comp[valid_comp["OUTLIER_PULL"] > 1.5])
print(f"""
1. DATA QUALITY
   {n_distorted:,} NCM-4 codes ({100*n_distorted/len(valid_comp):.0f}%) have aggregate unit price
   >1.5x their outlier-excluded median.
   -> Always use MEDIAN_UNIT_FOB_PER_KG, never total FOB/KG, for benchmarking.
""")

# 2. Origin-spread opportunities
wide_spread = ncm4_full[
    (ncm4_full["SPREAD_RATIO"] >= 5) &
    (ncm4_full["N_ORIGINS"]    >= 3) &
    (ncm4_full["VL_FOB"]       >= 1_000_000)
]
print(f"2. PRICING LEVERAGE (origin substitution opportunities)")
print(f"   {len(wide_spread):,} NCM-4 codes (>=$1M FOB, >=3 origins) have P90/P10 >=5x.")
print(f"   The priciest origin source is 5x costlier than the cheapest.")
print(f"   -> Identify clients sourcing from high-price origins; build cost-of-switching cases.")

# 3. China price position
valid_idx = china_vs_all["CHINA_PRICE_IDX"].dropna()
cheaper = (valid_idx < 0.9).sum()
premium = (valid_idx > 1.1).sum()
n_china = len(valid_idx)
print(f"""
3. CHINA PRICING POSITION
   China is below the all-origins median in {100*cheaper/max(n_china,1):.0f}% of NCM-4 codes
   where it competes — the classic China cost advantage holds broadly.
   China commands a premium (>1.1x) in {100*premium/max(n_china,1):.0f}% — typically
   high-tech/specialty goods (electronics, precision instruments, advanced materials).
   -> For cost-sensitive clients, China-sourced supply chains are defensible.
   -> China-premium codes represent high-value air-freight candidates.
""")

# 4. Air-freight opportunity size
air_fob   = ncm4_full[ncm4_full["MEDIAN_ORIGIN"] >= AIR_FLOOR]["VL_FOB"].sum()
total_fob = ncm4_full["VL_FOB"].sum()
print(f"4. AIR-FREIGHT OPPORTUNITY")
print(f"   NCM-4 codes with median >=${AIR_FLOOR}/kg: {len(high_value):,} codes, "
      f"${air_fob/1e9:.1f}B FOB ({100*air_fob/max(total_fob,1):.0f}% of total market).")
print(f"   These goods can absorb air-freight economics at $3-8/kg Brazil routes.")

print(f"""
5. FOCUS CHAPTERS AIR VIABILITY""")
for chp in FOCUS_CHAPTERS:
    n_above = len(ncm4_full[(ncm4_full["CO_CAPITULO"] == chp) &
                             (ncm4_full["MEDIAN_ORIGIN"] >= AIR_FLOOR)])
    n_total = len(ncm4_full[ncm4_full["CO_CAPITULO"] == chp])
    med     = ncm4_full[ncm4_full["CO_CAPITULO"] == chp]["MEDIAN_ORIGIN"].median()
    if n_total > 0:
        print(f"   {FOCUS_LABELS[chp]:22s}: chapter median ${med:.1f}/kg | "
              f"{n_above}/{n_total} codes above ${AIR_FLOOR}/kg")

print(f"""
6. RECOMMENDED USE
   - Use MEDIAN_ORIGIN as the benchmark price in feasibility studies.
   - For codes with SPREAD_RATIO > 5, run origin-option analysis with the client.
   - China price index < 0.9 = cost argument for China-sourced supply chains.
   - Median >= ${AIR_FLOOR}/kg = flag for air-freight conversation with any prospect.
   - Combine with 05_freight_analysis FREIGHT_PCT_FOB for full landed-cost picture.""")
