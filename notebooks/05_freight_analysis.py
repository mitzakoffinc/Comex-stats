# %% [markdown]
# # 05 — Freight Analysis
#
# **Purpose:** Identify which product segments and transport modes carry the highest freight cost burden.
# **Input:** `outputs/data/enriched.parquet`
# **Output:** Freight intensity rankings by HS chapter × mode — input to logistics sales prospecting.
#
# ---
#
# ### Dataset limitation — net weight vs. chargeable weight
#
# Freight carriers bill based on **chargeable weight** = `max(gross_weight, CBM / 6000)` for air;
# sea LCL uses CBM/1000 (wt/m³). This dataset only contains `KG_LIQUIDO` (net weight — product
# only, no tare/packaging). Consequently:
#
# | Metric | Limitation | Reliability |
# |---|---|---|
# | `FREIGHT_PER_KG` = VL_FRETE / KG_LIQUIDO | Net kg ≠ chargeable weight. Biased upward for light/bulky goods. | **Proxy only** — use for relative comparison, not absolute cost reconstruction |
# | `FREIGHT_PCT_FOB` = VL_FRETE / VL_FOB | Dimensionless ratio; unaffected by weight/volume gap | **More reliable** — preferred metric for freight intensity |
#
# All charts and tables in this notebook use `FREIGHT_PCT_FOB` as the primary metric and flag
# `FREIGHT_PER_KG` as a proxy.

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
CHARTS_DIR   = PROJECT_ROOT / "outputs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (12, 5)})

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

_enr_path = str(DATA_DIR / "enriched.parquet").replace("\\", "/")
_con = duckdb.connect()
enr = _con.execute(f"""
    SELECT
        CO_ANO, CO_CAPITULO, CO_PAIS, NO_PAIS_ING,
        CO_VIA, NO_VIA,
        KG_LIQUIDO, VL_FOB, VL_FRETE,
        FREIGHT_PER_KG, FREIGHT_PCT_FOB,
        is_outlier_unitval
    FROM read_parquet('{_enr_path}')
""").df()
print(f"Loaded {len(enr):,} rows, {len(enr.columns)} columns")
print(f"VL_FRETE populated: {enr['VL_FRETE'].notna().sum():,} rows ({100 * enr['VL_FRETE'].notna().mean():.1f}%)")

# %% [markdown]
# ## 1. Freight data completeness

# %%
# Overall VL_FRETE population by year
cov_year = (
    enr.groupby("CO_ANO")
    .apply(lambda g: pd.Series({
        "total_rows":       len(g),
        "frete_populated":  g["VL_FRETE"].notna().sum(),
        "frete_pct":        round(100 * g["VL_FRETE"].notna().mean(), 1),
    }), include_groups=False)
)
print("VL_FRETE population by year:")
display(cov_year)

# By transport mode
cov_via = (
    enr.groupby(["CO_VIA", "NO_VIA"])
    .apply(lambda g: pd.Series({
        "total_rows":      len(g),
        "frete_populated": g["VL_FRETE"].notna().sum(),
        "frete_pct":       round(100 * g["VL_FRETE"].notna().mean(), 1),
    }), include_groups=False)
    .sort_values("total_rows", ascending=False)
    .reset_index()
)
print("\nVL_FRETE population by transport mode:")
display(cov_via)

# %%
# Bar chart: freight coverage by transport mode
fig, ax = plt.subplots(figsize=(10, 4))
via_labels = cov_via["NO_VIA"].fillna("Unknown").tolist()
bars = ax.barh(via_labels, cov_via["frete_pct"], color="steelblue")
ax.bar_label(bars, fmt="%.0f%%", padding=4, fontsize=9)
ax.set_xlabel("% rows with VL_FRETE populated")
ax.set_title("VL_FRETE data coverage by transport mode")
ax.set_xlim(0, 115)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "freight_coverage_by_via.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Freight intensity by HS chapter — FREIGHT_PCT_FOB

# %%
# Restrict to rows with valid freight data and positive FOB
has_frete = enr[enr["VL_FRETE"].notna() & (enr["VL_FOB"] > 0)].copy()
print(f"Rows with VL_FRETE + VL_FOB > 0: {len(has_frete):,}")

# Median FREIGHT_PCT_FOB by chapter, weighted by total FOB
chap_freight = (
    has_frete.groupby("CO_CAPITULO")
    .agg(
        N_ROWS=("VL_FOB", "count"),
        VL_FOB_total=("VL_FOB", "sum"),
        VL_FRETE_total=("VL_FRETE", "sum"),
        MEDIAN_FRETE_PCT=("FREIGHT_PCT_FOB", "median"),
    )
    .reset_index()
)
# Aggregate freight intensity = total frete / total FOB (better than median of ratios for sizing)
chap_freight["AGG_FRETE_PCT"] = (
    chap_freight["VL_FRETE_total"] / chap_freight["VL_FOB_total"] * 100
).round(2)
chap_freight["VL_FOB_bn"] = (chap_freight["VL_FOB_total"] / 1e9).round(3)

top20_chap = chap_freight.nlargest(20, "AGG_FRETE_PCT")
print("\nTop 20 chapters by aggregate freight intensity (VL_FRETE / VL_FOB):")
display(top20_chap[["CO_CAPITULO", "N_ROWS", "VL_FOB_bn", "AGG_FRETE_PCT", "MEDIAN_FRETE_PCT"]].reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    top20_chap["CO_CAPITULO"].astype(str),
    top20_chap["AGG_FRETE_PCT"],
    color="tomato",
)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
ax.set_xlabel("Aggregate freight intensity — VL_FRETE / VL_FOB (%)")
ax.set_title("Top 20 HS chapters by freight intensity (% of FOB)")
ax.set_xlim(0, top20_chap["AGG_FRETE_PCT"].max() * 1.2)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "freight_intensity_by_chapter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Freight intensity by origin country (top 20 by VL_FOB)
#
# Dimensionless ratio — identifies which supplier countries incur the highest declared freight
# relative to cargo value.

# %%
country_freight = (
    has_frete.groupby(["CO_PAIS", "NO_PAIS_ING"])
    .agg(
        N_ROWS=("VL_FOB", "count"),
        VL_FOB_total=("VL_FOB", "sum"),
        VL_FRETE_total=("VL_FRETE", "sum"),
    )
    .reset_index()
)
country_freight["AGG_FRETE_PCT"] = (
    country_freight["VL_FRETE_total"] / country_freight["VL_FOB_total"] * 100
).round(2)
country_freight["VL_FOB_bn"] = (country_freight["VL_FOB_total"] / 1e9).round(3)

# Filter to top 20 origins by FOB value (meaningful markets only)
top20_fob_countries = country_freight.nlargest(20, "VL_FOB_total")["CO_PAIS"].tolist()
c_top = (
    country_freight[country_freight["CO_PAIS"].isin(top20_fob_countries)]
    .sort_values("AGG_FRETE_PCT", ascending=False)
)

print("Freight intensity — top 20 countries by VL_FOB, sorted by freight intensity:")
display(c_top[["NO_PAIS_ING", "VL_FOB_bn", "AGG_FRETE_PCT"]].reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(12, 6))
labels = c_top["NO_PAIS_ING"].fillna(c_top["CO_PAIS"]).tolist()
bars = ax.barh(labels, c_top["AGG_FRETE_PCT"], color="steelblue")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
ax.set_xlabel("Aggregate freight intensity — VL_FRETE / VL_FOB (%)")
ax.set_title("Freight intensity by origin country (top 20 countries by FOB value)")
ax.set_xlim(0, c_top["AGG_FRETE_PCT"].max() * 1.2)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(CHARTS_DIR / "freight_intensity_by_country.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Air vs. sea freight intensity
#
# Air freight is expected to show far higher `FREIGHT_PCT_FOB` than sea — this validates the
# signal. Chapters that appear in *both* high-air-intensity and high-sea-intensity lists are
# structurally freight-heavy regardless of mode.

# %%
via_summary = (
    has_frete.groupby(["CO_VIA", "NO_VIA"])
    .agg(
        N_ROWS=("VL_FOB", "count"),
        VL_FOB_bn=("VL_FOB", lambda x: round(x.sum() / 1e9, 3)),
        AGG_FRETE_PCT=("VL_FRETE", lambda x: round(x.sum() / has_frete.loc[x.index, "VL_FOB"].sum() * 100, 2)),
        MEDIAN_FRETE_PCT=("FREIGHT_PCT_FOB", lambda x: round(x.median() * 100, 2)),
    )
    .sort_values("N_ROWS", ascending=False)
    .reset_index()
)
print("Freight intensity by transport mode:")
display(via_summary)

# %%
# Chapter-level breakdown: air vs sea intensity side by side
# Identify air and sea codes from the data (NO_VIA text), falling back to the
# zero-padded SISCOMEX codes: '04' = AEREA, '01' = MARITIMA
if "NO_VIA" in has_frete.columns and has_frete["NO_VIA"].notna().any():
    air_codes  = has_frete[has_frete["NO_VIA"].str.lower().str.contains("a.?r|air", na=False, regex=True)]["CO_VIA"].unique().tolist()
    sea_codes  = has_frete[has_frete["NO_VIA"].str.lower().str.contains("mar|sea|aqu", na=False, regex=True)]["CO_VIA"].unique().tolist()
    print(f"Air CO_VIA codes detected: {air_codes}")
    print(f"Sea CO_VIA codes detected: {sea_codes}")
else:
    air_codes = ["04"]
    sea_codes = ["01"]
    print("NO_VIA not available — using default codes: air='04', sea='01'")

air_df = has_frete[has_frete["CO_VIA"].isin(air_codes)]
sea_df = has_frete[has_frete["CO_VIA"].isin(sea_codes)]

def chapter_intensity(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CO_CAPITULO").agg(
        VL_FOB_total=("VL_FOB", "sum"),
        VL_FRETE_total=("VL_FRETE", "sum"),
    ).reset_index()
    g["FRETE_PCT"] = (g["VL_FRETE_total"] / g["VL_FOB_total"] * 100).round(2)
    return g[["CO_CAPITULO", "VL_FOB_total", "FRETE_PCT"]]

air_chap = chapter_intensity(air_df).rename(columns={"FRETE_PCT": "AIR_FRETE_PCT", "VL_FOB_total": "VL_FOB_air"})
sea_chap = chapter_intensity(sea_df).rename(columns={"FRETE_PCT": "SEA_FRETE_PCT", "VL_FOB_total": "VL_FOB_sea"})

mode_comp = air_chap.merge(sea_chap, on="CO_CAPITULO", how="outer").fillna({"AIR_FRETE_PCT": 0, "SEA_FRETE_PCT": 0})
mode_comp["MAX_FRETE_PCT"] = mode_comp[["AIR_FRETE_PCT", "SEA_FRETE_PCT"]].max(axis=1)
top_mode = mode_comp.nlargest(15, "MAX_FRETE_PCT")

print("\nTop 15 chapters — air vs. sea freight intensity side by side:")
display(top_mode[["CO_CAPITULO", "AIR_FRETE_PCT", "SEA_FRETE_PCT"]].reset_index(drop=True))

# %%
# Grouped bar: air vs sea freight intensity
x = np.arange(len(top_mode))
w = 0.38
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, top_mode["AIR_FRETE_PCT"], w, label="Air",      color="#E84D3D")
ax.bar(x + w/2, top_mode["SEA_FRETE_PCT"], w, label="Maritime", color="#2E86AB")
ax.set_xticks(x)
ax.set_xticklabels([f"Ch {c}" for c in top_mode["CO_CAPITULO"].astype(str)], rotation=45, ha="right")
ax.set_ylabel("Freight intensity (% of FOB)")
ax.set_title("Air vs. maritime freight intensity — top 15 HS chapters")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(CHARTS_DIR / "freight_air_vs_sea_by_chapter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Freight per net kg (proxy) — high-variance chapters
#
# > **Proxy metric.** `FREIGHT_PER_KG = VL_FRETE / KG_LIQUIDO` uses **net weight only**. For
# > light, high-volume goods (electronics, clothing, auto parts), the chargeable weight from
# > volumetric measurement (`CBM / 6000`) will exceed net kg, so the true cost-per-kg is higher
# > than shown here. Use this metric only for relative comparisons between chapters — not for
# > absolute freight rate reconstruction.
#
# High variance in `FREIGHT_PER_KG` within a chapter signals mixed product density — some SKUs
# are compact/heavy (billed by weight), others are light/bulky (billed volumetrically).

# %%
# Only rows with valid weight and freight
has_frete_kg = has_frete[has_frete["KG_LIQUIDO"] > 0].copy()

def p10(s): return float(s.quantile(0.10)) if len(s) else np.nan
def p90(s): return float(s.quantile(0.90)) if len(s) else np.nan

chap_fkg = (
    has_frete_kg
    .loc[~has_frete_kg["is_outlier_unitval"]]
    .groupby("CO_CAPITULO")["FREIGHT_PER_KG"]
    .agg([
        ("MEDIAN_FRT_KG", "median"),
        ("P10_FRT_KG",    p10),
        ("P90_FRT_KG",    p90),
        ("N",             "count"),
    ])
    .reset_index()
)
chap_fkg["P10_P90_SPREAD"] = (chap_fkg["P90_FRT_KG"] - chap_fkg["P10_FRT_KG"]).round(4)
chap_fkg = chap_fkg.sort_values("MEDIAN_FRT_KG", ascending=False)

print("Top 20 chapters by median freight/net-kg (proxy — see note above):")
display(
    chap_fkg.head(20)[["CO_CAPITULO", "N", "MEDIAN_FRT_KG", "P10_FRT_KG", "P90_FRT_KG", "P10_P90_SPREAD"]]
    .round(4)
    .reset_index(drop=True)
)

# %% [markdown]
# ## 6. Priority segments — high freight intensity × meaningful volume
#
# Combining freight intensity (`FREIGHT_PCT_FOB`) with scale (`VL_FOB`) and mode identifies the
# best target segments for QEntrega/Itatibense freight optimization conversations.

# %%
# Chapter × mode matrix: aggregate freight intensity + volume
seg = (
    has_frete.groupby(["CO_CAPITULO", "CO_VIA", "NO_VIA"])
    .agg(
        N_OPS=("VL_FOB",    "count"),
        VL_FOB_total=("VL_FOB",    "sum"),
        VL_FRETE_total=("VL_FRETE", "sum"),
    )
    .reset_index()
)
seg["AGG_FRETE_PCT"] = (seg["VL_FRETE_total"] / seg["VL_FOB_total"] * 100).round(2)
seg["VL_FOB_mm"]     = (seg["VL_FOB_total"] / 1e6).round(2)  # USD millions

# Thresholds: sufficient volume AND meaningful freight intensity
N_OPS_MIN    = 500      # minimum operations
FOB_MIN_MM   = 50       # USD 50M+ FOB
FRETE_PCT_MIN = 5.0     # at least 5% freight intensity

priority = seg[
    (seg["N_OPS"]        >= N_OPS_MIN) &
    (seg["VL_FOB_mm"]    >= FOB_MIN_MM) &
    (seg["AGG_FRETE_PCT"] >= FRETE_PCT_MIN)
].sort_values("AGG_FRETE_PCT", ascending=False)

print(f"Priority segments (N_OPS >= {N_OPS_MIN}, FOB >= USD {FOB_MIN_MM}M, freight intensity >= {FRETE_PCT_MIN}%):")
print(f"Total: {len(priority)} segments")
display(
    priority[["CO_CAPITULO", "NO_VIA", "N_OPS", "VL_FOB_mm", "AGG_FRETE_PCT"]]
    .head(30)
    .reset_index(drop=True)
)

# %%
# Scatter: freight intensity vs. FOB scale, coloured by transport mode
if len(priority) > 1:
    mode_colors = {}
    palette = plt.cm.Set2.colors
    for i, via in enumerate(priority["NO_VIA"].unique()):
        mode_colors[via] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(11, 6))
    for via, grp in priority.groupby("NO_VIA"):
        ax.scatter(
            grp["VL_FOB_mm"],
            grp["AGG_FRETE_PCT"],
            label=via,
            s=grp["N_OPS"] / 10,
            alpha=0.7,
            color=mode_colors.get(via, "gray"),
        )
        for _, row in grp.iterrows():
            ax.annotate(
                f"Ch {row['CO_CAPITULO']}",
                (row["VL_FOB_mm"], row["AGG_FRETE_PCT"]),
                fontsize=7, alpha=0.8,
                xytext=(4, 2), textcoords="offset points",
            )

    ax.set_xlabel("Total FOB (USD millions)")
    ax.set_ylabel("Freight intensity (% of FOB)")
    ax.set_title("Priority segments: freight intensity vs. FOB scale\n(bubble size = N operations)")
    ax.legend(title="Mode", loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "freight_priority_scatter.png", bbox_inches="tight")
    plt.show()
else:
    print("Fewer than 2 priority segments — adjust thresholds (N_OPS_MIN, FOB_MIN_MM, FRETE_PCT_MIN) to see scatter.")

# %% [markdown]
# ## 7. Freight intensity — China-specific deep dive
#
# Chinese imports as a subset: confirms whether China lanes carry above- or below-average
# freight intensity relative to the market, by chapter.

# %%
# Resolve China code from data — no hardcoding
_cn = enr[enr["NO_PAIS_ING"].str.contains("China", case=False, na=False)]
CHINA_CODE = _cn["CO_PAIS"].iloc[0] if len(_cn) else "160"
print(f"China CO_PAIS resolved: {CHINA_CODE}")

china_frete = has_frete[has_frete["CO_PAIS"] == CHINA_CODE].copy()
print(f"China rows with VL_FRETE: {len(china_frete):,}")
print(f"China VL_FRETE coverage: {100 * len(china_frete) / len(enr[enr['CO_PAIS'] == CHINA_CODE]):.1f}%")

cn_chap = (
    china_frete.groupby("CO_CAPITULO")
    .agg(
        N_OPS=("VL_FOB", "count"),
        VL_FOB_mm=("VL_FOB", lambda x: round(x.sum() / 1e6, 2)),
        FRETE_total=("VL_FRETE", "sum"),
        FOB_total=("VL_FOB", "sum"),
    )
    .reset_index()
)
cn_chap["AGG_FRETE_PCT"] = (cn_chap["FRETE_total"] / cn_chap["FOB_total"] * 100).round(2)

# Merge with overall market intensity for comparison
market_intensity = chap_freight[["CO_CAPITULO", "AGG_FRETE_PCT"]].rename(
    columns={"AGG_FRETE_PCT": "MARKET_FRETE_PCT"}
)
cn_vs_market = cn_chap.merge(market_intensity, on="CO_CAPITULO", how="left")
cn_vs_market["CHINA_PREMIUM"] = (cn_vs_market["AGG_FRETE_PCT"] - cn_vs_market["MARKET_FRETE_PCT"]).round(2)

top20_cn = cn_vs_market.nlargest(20, "AGG_FRETE_PCT")
print("\nTop 20 chapters from China by freight intensity (vs. overall market):")
display(
    top20_cn[["CO_CAPITULO", "N_OPS", "VL_FOB_mm", "AGG_FRETE_PCT", "MARKET_FRETE_PCT", "CHINA_PREMIUM"]]
    .reset_index(drop=True)
)

# %%
# Diverging bar: China freight intensity vs. market average
plot_df = cn_vs_market[
    cn_vs_market["VL_FOB_mm"] >= 10  # at least USD 10M — avoid noise
].nlargest(15, "AGG_FRETE_PCT").reset_index(drop=True)

if len(plot_df) > 0:
    x = np.arange(len(plot_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, plot_df["AGG_FRETE_PCT"],    w, label="China",  color="#E84D3D")
    ax.bar(x + w/2, plot_df["MARKET_FRETE_PCT"], w, label="All origins", color="#AAAAAA")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ch {c}" for c in plot_df["CO_CAPITULO"].astype(str)], rotation=45, ha="right")
    ax.set_ylabel("Freight intensity (% of FOB)")
    ax.set_title("China vs. all-origins freight intensity by HS chapter\n(chapters with FOB >= USD 10M)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "freight_china_vs_market.png", bbox_inches="tight")
    plt.show()
else:
    print("No China chapters meet the VL_FOB >= USD 10M threshold — lower the filter or check data.")

# %% [markdown]
# ## 8. Methodology note — limitations and next steps
#
# ### What this analysis CAN do
# - Rank HS chapters and transport modes by **relative freight burden** (`FREIGHT_PCT_FOB`)
# - Identify chapters where declared freight cost is systematically high → freight-sensitive
#   cargo → highest optimization potential
# - Confirm the air/sea split intuition (air = higher `FREIGHT_PCT_FOB`)
# - Flag China-specific freight patterns vs. the overall import market
#
# ### What this analysis CANNOT do
# | Gap | Reason | Data needed to close |
# |---|---|---|
# | Reconstruct quoted freight rates | `KG_LIQUIDO` is net weight; chargeable weight = max(gross, CBM/6000) | Gross weight + volumetric dimensions per shipment |
# | Identify individual importers | Public Comex Stat is anonymized | DI-level data (DUIMP) or Logcomex/ImportGenius overlay |
# | Compute profit margins per lane | No carrier rate card in dataset | Carrier rate cards or forwarding quotes |
# | Validate air `FREIGHT_PCT_FOB` for light goods | Volumetric billing would increase real cost/kg above what `FREIGHT_PER_KG` shows | Same as above — gross weight and dimensions |
#
# ### Next step for freight cost reconstruction
# To move from intensity proxies to actual cost modeling:
# 1. **Obtain packing-list samples** for target chapters (e.g., electronics, auto parts, pharma)
#    → derive typical `gross_weight / net_weight` ratio and `CBM / gross_kg` ratio
# 2. Apply these empirical multipliers to `KG_LIQUIDO` to estimate chargeable weight
# 3. Cross-reference against ANAC (air) or ANTAQ (sea) published rate indices for cost benchmarks
#
# ---
#
# ## So what? — Segment targeting for QEntrega and Itatibense
#
# Use the priority segments table above (§6) with the following framing:
#
# **For QEntrega (air freight focus):**
# - Target chapters in the **top-right quadrant** of the scatter (high FOB volume + high freight
#   intensity via air)
# - These represent importers for whom freight cost is a significant line item — freight
#   optimization is a real conversation
# - Strongest prospects: pharma (Chapter 30), electronics sub-assemblies, high-value auto parts
#
# **For Itatibense Transportes (road/drayage, inland logistics):**
# - Focus on **maritime high-freight-intensity** chapters: cargo arriving at Santos/Itajaí with
#   high declared freight per FOB → importers who feel freight cost pain and may be receptive to
#   drayage + customs brokerage bundling
# - Chapter × URF combinations with high `AGG_FRETE_PCT` and large N_OPS are best cold-call targets
#
# **Data caveat:** Since public Comex Stat has no importer names, map the priority segments back
# to specific companies using Logcomex, ImportGenius, or Receita Federal DI-level data as a next step.
