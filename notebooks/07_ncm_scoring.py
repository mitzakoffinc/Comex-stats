# %% [markdown]
# # 07 — NCM Scoring: Attractiveness Ranking
#
# **Purpose:** Rank NCM-4 headings by composite attractiveness score for freight forwarding
# opportunities, combining 6 weighted criteria.
#
# **Inputs:**
# - `outputs/data/mart_ncm4_annual.parquet` — volume, growth, HHI, dominant mode/URF, n_months
# - `outputs/data/mart_sh6_via.parquet` — freight intensity by product × mode
# - `outputs/data/mart_ncm4_urf.parquet` — URF breakdown for route scoring
# - `outputs/data/mart_ncm4_month.parquet` — monthly series for partial-year momentum
# - `Config/config.xlsx` sheets `scoring_weights`, `scoring_thresholds`, `target_urfs`
#
# **Output:** Ranked top-N table + scatter plot (volume vs CAGR, bubble=HHI, color=modal match)
#
# **Methodology notes:**
# - The reference year is the **latest COMPLETE year** (12 months of data). Partial years
#   are never used for ranking — they feed only the YTD momentum column, computed on a
#   same-months basis (e.g. Jan–Mar partial year vs Jan–Mar reference year).
# - CAGR uses the longest available span of complete years up to `crescimento_cagr_anos`.
#
# **Scoring criteria (weights in config.xlsx):**
# 1. Volume — total FOB in the reference year
# 2. Crescimento — CAGR over complete years
# 3. Concentração — HHI URF (low HHI = fragmented = more opportunity)
# 4. Modal — % air share; high air share matches the operator's air-freight profile
# 5. Rota — % volume entering via target URFs (GRU / Viracopos / Santos, from config)
# 6. Margem de frete — median freight as % of FOB
#
# Run `pipeline/00_ingest.py → 01_clean.py → 02_enrich.py → 03_marts.py` before executing.

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Works both as a script (python notebooks/07_ncm_scoring.py) and cell-by-cell in
# VS Code's Interactive Window, where __file__ is undefined: walk upward from the
# best-known location until Config/config.xlsx is found.
def _find_project_root() -> Path:
    start = Path(globals().get("__file__", Path.cwd() / "x")).resolve().parent
    for p in [start, *start.parents]:
        if (p / "Config" / "config.xlsx").exists():
            return p
    raise FileNotFoundError("Project root not found (no Config/config.xlsx upward of cwd)")

PROJECT_ROOT = _find_project_root()
DATA_DIR     = PROJECT_ROOT / "outputs" / "data"
CHART_DIR    = PROJECT_ROOT / "outputs" / "charts"
CONFIG_XLSX  = PROJECT_ROOT / "Config" / "config.xlsx"
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

AIR_VIA_CODE = "04"  # CO_VIA '04' = AEREA (see Data/References/VIA.csv)

# %% [markdown]
# ## Parameters (from Config/config.xlsx)

# %%
_cfg = pd.read_excel(CONFIG_XLSX, sheet_name=None)

# Scoring weights (must sum to 1)
weights = _cfg["scoring_weights"].set_index("criterio")["peso"].to_dict()

# Scoring thresholds
thresh = _cfg["scoring_thresholds"].set_index("parametro")["valor"]
TOP_N           = int(thresh["top_n_ncms"])
VOL_MIN_FOB     = float(thresh["vol_min_fob_usd"])
CAGR_YEARS      = int(thresh["crescimento_cagr_anos"])
HHI_PULVERIZADO = float(thresh["hhi_pulverizado"])
HHI_CONCENTRADO = float(thresh["hhi_concentrado"])
FRETE_ALTO_PCT  = float(thresh["frete_alto_pct"])

# Target URFs for route scoring — maintained in config.xlsx sheet 'target_urfs'.
# Codes are 7-digit zero-padded strings (e.g. 0817600 = Aeroporto GRU); Excel may
# store them as numbers, so re-pad defensively.
if "target_urfs" not in _cfg:
    raise KeyError(
        "Sheet 'target_urfs' not found in Config/config.xlsx. "
        "Add it with columns ['co_urf', 'descricao'] (see README)."
    )
_urfs = _cfg["target_urfs"]
TARGET_URFS = [str(u).split(".")[0].zfill(7) for u in _urfs["co_urf"]]

print(f"Weights : {weights}")
print(f"Sum     : {sum(weights.values()):.2f}")
assert abs(sum(weights.values()) - 1.0) < 1e-6, "Scoring weights must sum to 1"
print(f"Top N   : {TOP_N}")
print(f"Vol min : USD {VOL_MIN_FOB:,.0f}")
print(f"CAGR yrs: {CAGR_YEARS} (max; actual span depends on complete years available)")
print("Target URFs:")
for code, desc in zip(TARGET_URFS, _urfs["descricao"]):
    print(f"  {code} — {desc}")

# %% [markdown]
# ## 1. Load Base Data

# %%
annual_path = str(DATA_DIR / "mart_ncm4_annual.parquet").replace("\\", "/")
via_path    = str(DATA_DIR / "mart_sh6_via.parquet").replace("\\", "/")
urf_path    = str(DATA_DIR / "mart_ncm4_urf.parquet").replace("\\", "/")
month_path  = str(DATA_DIR / "mart_ncm4_month.parquet").replace("\\", "/")

con = duckdb.connect()

annual = con.execute(f"SELECT * FROM read_parquet('{annual_path}')").df()
print(f"mart_ncm4_annual : {annual.shape[0]:,} rows, years {annual['CO_ANO'].min()}–{annual['CO_ANO'].max()}")
print(f"Distinct NCM-4   : {annual['CO_POSICAO'].nunique():,}")

# %% [markdown]
# ## 2. Determine Reference Year (latest COMPLETE year)
#
# Ranking on a partial year would bias every volume figure downward and wreck CAGR,
# so the reference year is the most recent year with 12 distinct months of data.
# The partial year (if any) is used only for the same-months momentum signal.

# %%
months_per_year = (
    annual.groupby("CO_ANO")["n_months"].max().sort_index()
)
complete_years = [int(y) for y, m in months_per_year.items() if m == 12]
partial_years  = {int(y): int(m) for y, m in months_per_year.items() if m < 12}

if not complete_years:
    raise RuntimeError(
        "No complete year (12 months) in the data — cannot rank. "
        "Add full-year IMP_*.csv files and re-run the pipeline."
    )

REF_YEAR = max(complete_years)

# CAGR base: prefer REF_YEAR - CAGR_YEARS; fall back to the earliest complete year
desired_base = REF_YEAR - CAGR_YEARS
BASE_YEAR    = desired_base if desired_base in complete_years else min(complete_years)
CAGR_SPAN    = REF_YEAR - BASE_YEAR  # actual span used in the exponent

# Partial year for momentum (only if newer than REF_YEAR)
PARTIAL_YEAR   = max((y for y in partial_years if y > REF_YEAR), default=None)
PARTIAL_MONTHS = partial_years.get(PARTIAL_YEAR)

print(f"Complete years   : {complete_years}")
print(f"Reference year   : {REF_YEAR}")
if CAGR_SPAN > 0:
    print(f"CAGR             : {BASE_YEAR} → {REF_YEAR} ({CAGR_SPAN}-year span)")
    if CAGR_SPAN < CAGR_YEARS:
        print(f"  NOTE: requested {CAGR_YEARS}y but only {CAGR_SPAN}y of complete data available.")
else:
    print("CAGR             : UNAVAILABLE — only one complete year. "
          "Growth criterion will be neutral (0.5) for all NCMs.")
if PARTIAL_YEAR:
    print(f"Momentum         : {PARTIAL_YEAR} YTD ({PARTIAL_MONTHS} months) vs same months of {REF_YEAR}")
else:
    print("Momentum         : no partial year newer than the reference year — column will be empty.")

ref  = annual[annual["CO_ANO"] == REF_YEAR].copy()
base = annual.loc[annual["CO_ANO"] == BASE_YEAR, ["CO_POSICAO", "total_fob"]].rename(
    columns={"total_fob": "fob_base"}
)

print(f"\nNCM-4 with data in {REF_YEAR}: {len(ref):,}")
print(f"NCM-4 with data in {BASE_YEAR}: {len(base):,}")

# %% [markdown]
# ## 3. Compute Scoring Criteria

# %%
# ── Merge base year for CAGR ──────────────────────────────────────────────────
df = ref.merge(base, on="CO_POSICAO", how="left")

# Minimum volume filter
df = df[df["total_fob"] >= VOL_MIN_FOB].copy()
print(f"NCM-4 after volume filter (>= USD {VOL_MIN_FOB:,.0f}): {len(df):,}")

# ── Criterion 1: Volume ───────────────────────────────────────────────────────
# Log scale: higher FOB = higher score (percentile rank 0–1)
df["log_fob"] = np.log1p(df["total_fob"])
df["score_volume"] = df["log_fob"].rank(pct=True)

# ── Criterion 2: Crescimento (CAGR over complete years) ──────────────────────
if CAGR_SPAN > 0:
    df["cagr"] = np.where(
        (df["fob_base"] > 0) & df["fob_base"].notna(),
        (df["total_fob"] / df["fob_base"]) ** (1 / CAGR_SPAN) - 1,
        np.nan,
    )
    # Clip extreme CAGR at 5th/95th before ranking to reduce distortion
    p5, p95 = df["cagr"].quantile(0.05), df["cagr"].quantile(0.95)
    df["cagr_clipped"]      = df["cagr"].clip(p5, p95)
    df["score_crescimento"] = df["cagr_clipped"].rank(pct=True)
else:
    df["cagr"] = np.nan
    df["score_crescimento"] = np.nan  # neutral-filled in the composite

# ── Criterion 3: Concentração (HHI URF) ──────────────────────────────────────
# Low HHI = fragmented market = more opportunity → invert
df["score_concentracao"] = 1 - df["hhi_urf"].rank(pct=True)

print("Criteria 1–3 computed.")
df[["CO_POSICAO", "total_fob", "cagr", "hhi_urf",
    "score_volume", "score_crescimento", "score_concentracao"]].head()

# %%
# ── Criterion 4: Modal (air share) ───────────────────────────────────────────
# Use mart_sh6_via aggregated to CO_POSICAO (first 4 chars of CO_SH6)
via = con.execute(f"""
    SELECT
        LEFT(CO_SH6, 4)              AS CO_POSICAO,
        SUM(VL_FOB)                  AS fob_total,
        SUM(CASE WHEN CO_VIA = '{AIR_VIA_CODE}' THEN VL_FOB ELSE 0 END) AS fob_air
    FROM read_parquet('{via_path}')
    WHERE CO_SH6 IS NOT NULL
    GROUP BY LEFT(CO_SH6, 4)
""").df()

via["air_share"] = via["fob_air"] / via["fob_total"].replace(0, np.nan)
df = df.merge(via[["CO_POSICAO", "air_share"]], on="CO_POSICAO", how="left")

# Sanity check: if air share is zero everywhere, the VIA code mapping is broken
if (via["fob_air"].sum() or 0) == 0:
    raise RuntimeError(
        f"No FOB matched CO_VIA = '{AIR_VIA_CODE}' — check VIA codes in the mart "
        "(expected zero-padded strings like '04')."
    )

# Score: higher air share = higher value per kg = better margin for air freight forwarders
df["score_modal"] = df["air_share"].rank(pct=True)

print(f"Air share data joined for {via['CO_POSICAO'].nunique():,} NCM-4 headings")
print(f"Overall air share of FOB: {100 * via['fob_air'].sum() / via['fob_total'].sum():.1f}%")

# ── Criterion 5: Rota (target URF share) ─────────────────────────────────────
target_urfs_sql = ", ".join(f"'{u}'" for u in TARGET_URFS)
urf_df = con.execute(f"""
    SELECT
        CO_POSICAO,
        SUM(VL_FOB)                                                         AS fob_total_urf,
        SUM(CASE WHEN CO_URF IN ({target_urfs_sql}) THEN VL_FOB ELSE 0 END) AS fob_target_urf
    FROM read_parquet('{urf_path}')
    GROUP BY CO_POSICAO
""").df()

if (urf_df["fob_target_urf"].sum() or 0) == 0:
    raise RuntimeError(
        f"No FOB matched the target URFs {TARGET_URFS} — codes in config.xlsx "
        "sheet 'target_urfs' must be 7-digit zero-padded strings matching URF.csv."
    )

urf_df["target_urf_share"] = urf_df["fob_target_urf"] / urf_df["fob_total_urf"].replace(0, np.nan)
df = df.merge(urf_df[["CO_POSICAO", "target_urf_share"]], on="CO_POSICAO", how="left")
df["score_rota"] = df["target_urf_share"].rank(pct=True)

print(f"URF data joined. Target URF share of total FOB: "
      f"{100 * urf_df['fob_target_urf'].sum() / urf_df['fob_total_urf'].sum():.1f}%")

# %%
# ── Criterion 6: Margem de frete ─────────────────────────────────────────────
# median_freight_pct from mart_ncm4_annual (reference year row)
df["score_margem_frete"] = df["median_freight_pct"].rank(pct=True)

# ── Composite score ───────────────────────────────────────────────────────────
score_cols = {
    "volume"       : "score_volume",
    "crescimento"  : "score_crescimento",
    "concentracao" : "score_concentracao",
    "modal"        : "score_modal",
    "rota"         : "score_rota",
    "margem_frete" : "score_margem_frete",
}

df["score_total"] = sum(
    weights.get(criterio, 0) * df[col].fillna(0.5)  # neutral fill for missing
    for criterio, col in score_cols.items()
)

df["rank"] = df["score_total"].rank(ascending=False, method="min").astype(int)
df_ranked  = df.sort_values("score_total", ascending=False).reset_index(drop=True)

print(f"Scoring complete on {REF_YEAR} (complete year).")
print(f"  Max score : {df_ranked['score_total'].max():.3f}")
print(f"  Min score : {df_ranked['score_total'].min():.3f}")

# %% [markdown]
# ## 4. Partial-Year Momentum (context column, not scored)
#
# Same-months YTD comparison: partial-year FOB vs the same months of the reference year.
# This shows which ranked NCMs are accelerating or decelerating right now, without
# letting extrapolation error contaminate the ranking itself.

# %%
if PARTIAL_YEAR:
    momentum = con.execute(f"""
        WITH partial_months AS (
            SELECT DISTINCT CO_MES
            FROM read_parquet('{month_path}')
            WHERE CO_ANO = {PARTIAL_YEAR}
        ),
        ytd AS (
            SELECT
                CO_POSICAO,
                SUM(CASE WHEN CO_ANO = {PARTIAL_YEAR} THEN VL_FOB ELSE 0 END) AS fob_partial,
                SUM(CASE WHEN CO_ANO = {REF_YEAR}     THEN VL_FOB ELSE 0 END) AS fob_ref_same_months
            FROM read_parquet('{month_path}')
            WHERE CO_MES IN (SELECT CO_MES FROM partial_months)
              AND CO_ANO IN ({PARTIAL_YEAR}, {REF_YEAR})
            GROUP BY CO_POSICAO
        )
        SELECT
            CO_POSICAO,
            fob_partial,
            fob_ref_same_months,
            CASE WHEN fob_ref_same_months > 0
                 THEN fob_partial / fob_ref_same_months - 1
            END AS momentum_yoy
        FROM ytd
    """).df()

    df_ranked = df_ranked.merge(
        momentum[["CO_POSICAO", "momentum_yoy"]], on="CO_POSICAO", how="left"
    )
    print(f"Momentum computed: {PARTIAL_YEAR} months 1–{PARTIAL_MONTHS} vs same months {REF_YEAR}")
else:
    df_ranked["momentum_yoy"] = np.nan
    print("No partial year — momentum column left empty.")

# %% [markdown]
# ## 5. Top-N Ranking Table

# %%
display_cols = [
    "rank", "CO_POSICAO",
    "total_fob", "cagr", "momentum_yoy",
    "hhi_urf", "air_share", "target_urf_share", "median_freight_pct",
    "score_volume", "score_crescimento", "score_concentracao",
    "score_modal", "score_rota", "score_margem_frete",
    "score_total",
]

top_n = df_ranked.head(TOP_N)[display_cols].copy()

# Formatting helpers
fmt_pct = lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
top_n["total_fob"]          = top_n["total_fob"].map("${:,.0f}".format)
top_n["cagr"]               = top_n["cagr"].map(fmt_pct)
top_n["momentum_yoy"]       = top_n["momentum_yoy"].map(fmt_pct)
top_n["hhi_urf"]            = top_n["hhi_urf"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
top_n["air_share"]          = top_n["air_share"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
top_n["target_urf_share"]   = top_n["target_urf_share"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
top_n["median_freight_pct"] = top_n["median_freight_pct"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
for sc in ["score_volume","score_crescimento","score_concentracao",
           "score_modal","score_rota","score_margem_frete","score_total"]:
    top_n[sc] = top_n[sc].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

momentum_label = f"YTD{PARTIAL_YEAR}" if PARTIAL_YEAR else "YTD"
top_n.columns = [
    "#", "NCM-4",
    f"FOB {REF_YEAR}", f"CAGR {CAGR_SPAN}a", momentum_label,
    "HHI-URF", "Air%", "TargetURF%", "Frete%FOB",
    "S.Vol", "S.Cresc", "S.Conc", "S.Modal", "S.Rota", "S.Frete",
    "SCORE",
]

print(top_n.to_string(index=False))

# %% [markdown]
# ## 6. Scatter Plot: Volume × Growth × HHI × Modal

# %%
plot_df = df_ranked.head(TOP_N * 2).copy()  # slightly wider pool for the chart

# When CAGR is unavailable (single complete year), plot momentum on the y-axis instead
y_col, y_label = ("cagr", f"{CAGR_SPAN}-year CAGR")
if plot_df["cagr"].isna().all() and PARTIAL_YEAR:
    y_col, y_label = ("momentum_yoy", f"YTD {PARTIAL_YEAR} momentum (same months vs {REF_YEAR})")

plot_df = plot_df.dropna(subset=[y_col, "total_fob"])

# Bubble size: inverse HHI (more fragmented = bigger bubble)
max_hhi  = plot_df["hhi_urf"].fillna(HHI_CONCENTRADO).clip(upper=10000)
bub_size = (10000 - max_hhi.fillna(10000)) / 80 + 20  # range ~20–145

# Color: high air share = orange, else blue
AIR_THRESHOLD = 0.30
colors = [
    "#E07B39" if (not pd.isna(a) and a >= AIR_THRESHOLD) else "#4A7FB5"
    for a in plot_df["air_share"]
]

fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(
    np.log10(plot_df["total_fob"]),
    plot_df[y_col] * 100,
    s=bub_size,
    c=colors,
    alpha=0.7,
    edgecolors="white",
    linewidths=0.5,
)

# Label top N by score
for _, row in df_ranked.head(TOP_N).iterrows():
    if pd.isna(row[y_col]) or pd.isna(row["total_fob"]):
        continue
    ax.annotate(
        row["CO_POSICAO"],
        (np.log10(row["total_fob"]), row[y_col] * 100),
        fontsize=6.5,
        textcoords="offset points",
        xytext=(4, 3),
        color="#333333",
    )

ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"$10^{{{x:.0f}}}$"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E07B39', markersize=9,
           label=f'Air share ≥ {AIR_THRESHOLD*100:.0f}%'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#4A7FB5', markersize=9,
           label=f'Air share < {AIR_THRESHOLD*100:.0f}%'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#888888', markersize=5,
           label='Bubble size ∝ market fragmentation (1/HHI)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

ax.set_xlabel(f"FOB {REF_YEAR} (log scale, USD)", fontsize=10)
ax.set_ylabel(y_label, fontsize=10)
ax.set_title(f"NCM-4 Attractiveness — Volume × Growth × HHI × Modal ({REF_YEAR})", fontsize=12)

plt.tight_layout()
out = CHART_DIR / "07_ncm_scoring_scatter.png"
fig.savefig(out, bbox_inches="tight")
plt.show()
print(f"Chart saved → {out}")

# %% [markdown]
# ## 7. Score Component Breakdown (top-10 heatmap)

# %%
score_raw_cols = [
    "score_volume", "score_crescimento", "score_concentracao",
    "score_modal", "score_rota", "score_margem_frete",
]
heat = df_ranked.head(10).set_index("CO_POSICAO")[score_raw_cols].astype(float)
heat.columns = ["Volume", "Crescimento", "Concentração", "Modal", "Rota", "Margem Frete"]

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(heat.fillna(0.5).values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, fontsize=9)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index, fontsize=9)

for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        val = heat.values[i, j]
        if pd.isna(val):
            ax.text(j, i, "n/a", ha="center", va="center", color="black", fontsize=8)
        else:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if 0.25 < val < 0.75 else "white", fontsize=8)

plt.colorbar(im, ax=ax, label="Percentile score (0–1)")
ax.set_title("Score Components — Top 10 NCM-4", fontsize=11)
plt.tight_layout()
out2 = CHART_DIR / "07_ncm_scoring_heatmap.png"
fig.savefig(out2, bbox_inches="tight")
plt.show()
print(f"Chart saved → {out2}")

# %% [markdown]
# ## 8. Strategic Read-out

# %%
top10 = df_ranked.head(10)

high_air   = df_ranked[df_ranked["air_share"] >= AIR_THRESHOLD].head(5)["CO_POSICAO"].tolist()
high_frete = df_ranked[df_ranked["median_freight_pct"] >= FRETE_ALTO_PCT].head(5)["CO_POSICAO"].tolist()
low_hhi    = df_ranked[df_ranked["hhi_urf"] < HHI_PULVERIZADO].head(5)["CO_POSICAO"].tolist()
accel      = df_ranked[df_ranked["momentum_yoy"] > 0].head(5)["CO_POSICAO"].tolist()

print("=" * 60)
print(f"  NCM SCORING — ranked on {REF_YEAR} (complete year), top {TOP_N}")
print("=" * 60)
print(f"\nTop 10 NCM-4 by composite score:")
for _, row in top10.iterrows():
    cagr_str = f"{row['cagr']*100:+.1f}%" if pd.notna(row['cagr']) else "N/A"
    mom_str  = f"{row['momentum_yoy']*100:+.1f}%" if pd.notna(row['momentum_yoy']) else "N/A"
    air_str  = f"{row['air_share']*100:.0f}%" if pd.notna(row['air_share']) else "N/A"
    print(f"  {int(row['rank']):>3}. {row['CO_POSICAO']}  "
          f"score={row['score_total']:.3f}  "
          f"FOB=${row['total_fob']/1e6:.1f}M  "
          f"CAGR={cagr_str}  "
          f"YTD={mom_str}  "
          f"Air={air_str}")

print(f"\nHigh air share (≥{AIR_THRESHOLD*100:.0f}%) in top pool : {high_air}")
print(f"High freight margin (≥{FRETE_ALTO_PCT*100:.0f}%) in top pool : {high_frete}")
print(f"Fragmented markets (HHI < {HHI_PULVERIZADO:.0f}) in top pool : {low_hhi}")
if PARTIAL_YEAR:
    print(f"Accelerating YTD {PARTIAL_YEAR} in top pool : {accel}")
print(f"\nWeights applied: {weights}")
print("\n→ Adjust weights, thresholds, or target URFs in Config/config.xlsx and re-run.")
