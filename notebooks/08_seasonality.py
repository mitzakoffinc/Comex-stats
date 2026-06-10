# %% [markdown]
# # 08 — Seasonality Analysis
#
# **Purpose:** Visualise intra-year seasonality patterns for selected HS chapters and years.
#
# **Configuration:** All parameters (years and chapters to analyse) are read from
# `Config/config.xlsx` sheet `seasonality`. Edit the Excel file and re-run — no code changes
# needed. Set `anos` to `auto` to analyse every year present in the data.
#
# **Input:** `outputs/data/mart_ncm4_month.parquet`
#
# **Reliability note:** Seasonality conclusions need >= 24 months (2+ complete years) to be
# trustworthy. With a single complete year the charts show *shape*, not a confirmed pattern.
#
# **Sections:**
# 1. Monthly volume index (each year normalised to its own annual mean = 100)
# 2. Absolute month-over-month overlay (all selected years on the same axes)
# 3. Peak / trough detection per chapter
# 4. YoY same-month comparison for adjacent year pairs

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
CHART_DIR    = PROJECT_ROOT / "outputs" / "charts"
CONFIG_XLSX  = PROJECT_ROOT / "Config" / "config.xlsx"
CHART_DIR.mkdir(parents=True, exist_ok=True)

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# %% [markdown]
# ## Parameters (from Config/config.xlsx)

# %%
_cfg  = pd.read_excel(CONFIG_XLSX, sheet_name="seasonality").set_index("parametro")["valor"]

mart_path = str(DATA_DIR / "mart_ncm4_month.parquet").replace("\\", "/")
con = duckdb.connect()

_anos_raw = str(_cfg["anos"]).strip()
if _anos_raw.lower() == "auto":
    YEARS_TO_ANALYZE = [
        int(r[0]) for r in con.execute(
            f"SELECT DISTINCT CO_ANO FROM read_parquet('{mart_path}') ORDER BY CO_ANO"
        ).fetchall()
    ]
else:
    YEARS_TO_ANALYZE = [int(y) for y in _anos_raw.split(";")]

FOCUS_CHAPTERS = [c.strip().zfill(2) for c in str(_cfg["capitulos_foco"]).split(";")]

print(f"Years    : {YEARS_TO_ANALYZE}")
print(f"Chapters : {FOCUS_CHAPTERS}")
print()
print("To change these values, edit Config/config.xlsx sheet 'seasonality' and re-run.")
print("Set anos = auto to analyse every year present in the data.")

# %% [markdown]
# ## Load Data

# %%
years_sql    = ", ".join(str(y) for y in YEARS_TO_ANALYZE)
chapters_sql = ", ".join(f"'{c}'" for c in FOCUS_CHAPTERS)

# Aggregate from NCM-4 to chapter level; CO_CAPITULO derived from CO_POSICAO
df = con.execute(f"""
    SELECT
        LEFT(CO_POSICAO, 2) AS CO_CAPITULO,
        CO_ANO,
        CO_MES,
        SUM(VL_FOB)     AS VL_FOB,
        SUM(KG_LIQUIDO) AS KG_LIQUIDO,
        SUM(N_OPS)      AS N_OPS
    FROM read_parquet('{mart_path}')
    WHERE CO_ANO IN ({years_sql})
      AND LEFT(CO_POSICAO, 2) IN ({chapters_sql})
    GROUP BY LEFT(CO_POSICAO, 2), CO_ANO, CO_MES
    ORDER BY CO_CAPITULO, CO_ANO, CO_MES
""").df()

df["CO_ANO"] = df["CO_ANO"].astype(int)
df["CO_MES"] = df["CO_MES"].astype(int)

print(f"Loaded {len(df):,} rows")
print(f"Chapters present : {sorted(df['CO_CAPITULO'].unique())}")
print(f"Years present    : {sorted(df['CO_ANO'].unique())}")

# Warn if any requested chapter / year combination is missing
for cap in FOCUS_CHAPTERS:
    for yr in YEARS_TO_ANALYZE:
        n = len(df[(df["CO_CAPITULO"] == cap) & (df["CO_ANO"] == yr)])
        if n < 12:
            print(f"  WARNING: chapter {cap} year {yr} has only {n} months of data")

n_complete_years = (
    df.groupby("CO_ANO")["CO_MES"].nunique().eq(12).sum()
)
if n_complete_years < 2:
    print("\nWARNING: fewer than 2 complete years available — treat any seasonality")
    print("pattern below as indicative only, not confirmed.")

# %% [markdown]
# ## 1. Monthly Volume Index
#
# Each year is normalised to its own annual mean (= 100).
# The index isolates the *shape* of seasonality, removing year-over-year level differences.
# Values > 100 indicate above-average months; < 100 indicate below-average months.

# %%
# Compute monthly index: (monthly_fob / annual_mean) * 100
annual_mean = (
    df.groupby(["CO_CAPITULO", "CO_ANO"])["VL_FOB"]
    .mean()
    .rename("annual_mean")
    .reset_index()
)
idx = df.merge(annual_mean, on=["CO_CAPITULO", "CO_ANO"])
idx["vol_index"] = idx["VL_FOB"] / idx["annual_mean"] * 100

colors = plt.cm.tab10.colors
n_chapters = len(FOCUS_CHAPTERS)

fig, axes = plt.subplots(1, n_chapters, figsize=(5 * n_chapters, 4), sharey=False)
if n_chapters == 1:
    axes = [axes]

for ax, cap in zip(axes, FOCUS_CHAPTERS):
    sub = idx[idx["CO_CAPITULO"] == cap]
    for i, yr in enumerate(YEARS_TO_ANALYZE):
        yr_data = sub[sub["CO_ANO"] == yr].sort_values("CO_MES")
        if yr_data.empty:
            continue
        ax.plot(
            yr_data["CO_MES"], yr_data["vol_index"],
            marker="o", markersize=4,
            color=colors[i % len(colors)],
            label=str(yr),
        )
    ax.axhline(100, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.set_title(f"Chapter {cap}", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=7, rotation=45)
    ax.set_ylabel("Volume index (annual mean = 100)")
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Monthly Volume Index by Chapter", fontsize=13, y=1.02)
plt.tight_layout()
out = CHART_DIR / "08_seasonality_index.png"
fig.savefig(out, bbox_inches="tight")
plt.show()
print(f"Chart saved: {out}")

# %% [markdown]
# ## 2. Absolute Month-over-Month Overlay
#
# All selected years plotted on the same axes (absolute USD FOB).
# Use this to see both seasonality shape *and* year-over-year level shifts.

# %%
fig, axes = plt.subplots(1, n_chapters, figsize=(5 * n_chapters, 4), sharey=False)
if n_chapters == 1:
    axes = [axes]

for ax, cap in zip(axes, FOCUS_CHAPTERS):
    sub = df[df["CO_CAPITULO"] == cap]
    for i, yr in enumerate(YEARS_TO_ANALYZE):
        yr_data = sub[sub["CO_ANO"] == yr].sort_values("CO_MES")
        if yr_data.empty:
            continue
        ax.plot(
            yr_data["CO_MES"], yr_data["VL_FOB"] / 1e6,
            marker="o", markersize=4,
            color=colors[i % len(colors)],
            label=str(yr),
        )
    ax.set_title(f"Chapter {cap}", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=7, rotation=45)
    ax.set_ylabel("FOB (USD million)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Monthly FOB by Chapter — Year Overlay", fontsize=13, y=1.02)
plt.tight_layout()
out2 = CHART_DIR / "08_seasonality_overlay.png"
fig.savefig(out2, bbox_inches="tight")
plt.show()
print(f"Chart saved: {out2}")

# %% [markdown]
# ## 3. Peak / Trough Detection
#
# Identify the month with the highest (peak) and lowest (trough) volume for each chapter × year,
# then summarise how consistent the pattern is across years.

# %%
records = []
for cap in FOCUS_CHAPTERS:
    sub = df[df["CO_CAPITULO"] == cap]
    for yr in YEARS_TO_ANALYZE:
        yr_data = sub[sub["CO_ANO"] == yr]
        if len(yr_data) < 6:
            continue
        peak_row  = yr_data.loc[yr_data["VL_FOB"].idxmax()]
        trough_row = yr_data.loc[yr_data["VL_FOB"].idxmin()]
        records.append({
            "CO_CAPITULO" : cap,
            "CO_ANO"      : yr,
            "peak_month"  : int(peak_row["CO_MES"]),
            "peak_fob"    : peak_row["VL_FOB"],
            "trough_month": int(trough_row["CO_MES"]),
            "trough_fob"  : trough_row["VL_FOB"],
        })

peaks = pd.DataFrame(records)

pt = peaks.copy()
pt["peak_label"]   = pt["peak_month"].apply(lambda m: MONTH_LABELS[m - 1])
pt["trough_label"] = pt["trough_month"].apply(lambda m: MONTH_LABELS[m - 1])
pt["peak_fob"]     = pt["peak_fob"].map("${:,.0f}".format)
pt["trough_fob"]   = pt["trough_fob"].map("${:,.0f}".format)
display(pt[["CO_CAPITULO","CO_ANO","peak_label","peak_fob","trough_label","trough_fob"]]
        .rename(columns={"CO_CAPITULO":"Chapter","CO_ANO":"Year",
                         "peak_label":"Peak month","peak_fob":"Peak FOB",
                         "trough_label":"Trough month","trough_fob":"Trough FOB"})
        .reset_index(drop=True))

# Summary: most common peak / trough month per chapter
print("\nConsistency summary:")
for cap in FOCUS_CHAPTERS:
    sub_p = peaks[peaks["CO_CAPITULO"] == cap]
    if sub_p.empty:
        continue
    common_peak   = sub_p["peak_month"].mode().iloc[0]
    peak_count    = (sub_p["peak_month"] == common_peak).sum()
    common_trough = sub_p["trough_month"].mode().iloc[0]
    trough_count  = (sub_p["trough_month"] == common_trough).sum()
    print(
        f"  Chapter {cap}: "
        f"peak = {MONTH_LABELS[common_peak-1]} ({peak_count}/{len(sub_p)} yrs)  |  "
        f"trough = {MONTH_LABELS[common_trough-1]} ({trough_count}/{len(sub_p)} yrs)"
    )

# %% [markdown]
# ## 4. YoY Same-Month Comparison
#
# For each adjacent year pair (e.g. 2024→2025), show the month-by-month percentage change.
# A positive bar means that month grew YoY; negative means it shrank. Months missing in either
# year are left blank (not zero).

# %%
year_pairs = list(zip(YEARS_TO_ANALYZE[:-1], YEARS_TO_ANALYZE[1:]))

if not year_pairs:
    print("Need at least 2 years to compute YoY comparison.")
else:
    n_pairs = len(year_pairs)
    fig, axes = plt.subplots(
        n_chapters, n_pairs,
        figsize=(4 * n_pairs, 3.5 * n_chapters),
        sharey=False, squeeze=False,
    )

    for row, cap in enumerate(FOCUS_CHAPTERS):
        sub = df[df["CO_CAPITULO"] == cap]
        for col, (yr_a, yr_b) in enumerate(year_pairs):
            ax = axes[row][col]
            a = sub[sub["CO_ANO"] == yr_a].set_index("CO_MES")["VL_FOB"]
            b = sub[sub["CO_ANO"] == yr_b].set_index("CO_MES")["VL_FOB"]
            pct_change = ((b - a) / a.replace(0, np.nan) * 100).reindex(range(1, 13))

            bar_colors = [
                "#4A7FB5" if (not pd.isna(v) and v >= 0) else "#C94040"
                for v in pct_change
            ]
            # NaN months (missing in either year) stay blank — fillna(0) would fake a flat month
            ax.bar(range(1, 13), pct_change.values, color=bar_colors, width=0.7)
            ax.axhline(0, color="#aaaaaa", linewidth=0.8)
            ax.set_title(f"Ch {cap}: {yr_a} vs {yr_b}", fontsize=9)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(MONTH_LABELS, fontsize=6, rotation=45)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%")
            )

    fig.suptitle("YoY Same-Month Change (%) by Chapter", fontsize=12)
    plt.tight_layout()
    out3 = CHART_DIR / "08_seasonality_yoy.png"
    fig.savefig(out3, bbox_inches="tight")
    plt.show()
    print(f"Chart saved: {out3}")
