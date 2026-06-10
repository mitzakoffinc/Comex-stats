# %% [markdown]
# # 02 — Market Sizing
#
# **Purpose:** Quantify the overall Brazilian import market by chapter and NCM-4, identify
# concentration, and flag high-growth chapters relevant to QEntrega and Itatibense.
#
# **Input:** `outputs/data/enriched.parquet`, `outputs/data/mart_chapter_month.parquet`,
# `outputs/data/mart_ncm4_country.parquet`
#
# **Output:** Charts in `outputs/charts/`, chapter-level summary table.
#
# **Scope:** Rankings use the latest COMPLETE year. YoY compares the latest complete year
# against the partial year annualized — clearly labeled as an estimate.
#
# **Core QEntrega/Itatibense chapters (pharma/chem/hazmat):** 28, 29, 30, 38, 39

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
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (14, 5)})

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

FOCUS_CHAPTERS = ["28", "29", "30", "38", "39"]
FOCUS_LABELS = {
    "28": "Ch28 Inorganic chem",
    "29": "Ch29 Organic chem",
    "30": "Ch30 Pharma",
    "38": "Ch38 Misc chem",
    "39": "Ch39 Plastics",
}

# %%
_con    = duckdb.connect()
_enr_path = str(DATA_DIR / "enriched.parquet").replace("\\", "/")

enriched = _con.execute(f"""
    SELECT CO_ANO, CO_CAPITULO, CO_POSICAO, NO_NCM_POR,
           CO_URF, NO_URF, KG_LIQUIDO, VL_FOB
    FROM read_parquet('{_enr_path}')
""").df()
mart_cm = pd.read_parquet(DATA_DIR / "mart_chapter_month.parquet")
mart_nc = pd.read_parquet(DATA_DIR / "mart_ncm4_country.parquet")

print(f"enriched          : {len(enriched):,} rows")
print(f"mart_chapter_month: {len(mart_cm):,} rows")
print(f"mart_ncm4_country : {len(mart_nc):,} rows")

# %%
# Dynamic period detection: rank on the latest COMPLETE year; the most recent
# partial year (if any) feeds the annualized YoY estimate.
months_per_year = mart_cm.groupby("CO_ANO")["CO_MES"].nunique().sort_index()
complete_years  = [int(y) for y, m in months_per_year.items() if m == 12]
if not complete_years:
    raise RuntimeError("No complete year (12 months) in the data — cannot build rankings.")

REF_YEAR = max(complete_years)
partials = {int(y): int(m) for y, m in months_per_year.items() if m < 12 and y > REF_YEAR}
PARTIAL_YEAR   = max(partials) if partials else None
PARTIAL_MONTHS = partials.get(PARTIAL_YEAR)

print(f"Complete years : {complete_years}")
print(f"Reference year : {REF_YEAR}")
if PARTIAL_YEAR:
    print(f"Partial year   : {PARTIAL_YEAR} ({PARTIAL_MONTHS} months) — used only for annualized YoY")

# %% [markdown]
# ## 1. Total market snapshot — all years

# %%
annual = enriched.groupby("CO_ANO")[["KG_LIQUIDO", "VL_FOB"]].sum()
annual["N_OPS"] = enriched.groupby("CO_ANO").size()

annual["VL_FOB_bn"] = (annual["VL_FOB"] / 1e9).round(2)
annual["KG_MM_mt"]  = (annual["KG_LIQUIDO"] / 1e9).round(2)
annual["N_OPS_k"]   = (annual["N_OPS"] / 1e3).round(1)

print("Annual market totals (FOB USD billions | million metric tons | k operations):")
display(annual[["VL_FOB_bn", "KG_MM_mt", "N_OPS_k"]].rename(
    columns={"VL_FOB_bn": "FOB USD bn", "KG_MM_mt": "MMt", "N_OPS_k": "Ops (k)"})
)
if PARTIAL_YEAR:
    print(f"\nNote: {PARTIAL_YEAR} = partial year ({PARTIAL_MONTHS} months). "
          "Do not compare raw totals without annualizing.")

# %% [markdown]
# ## 2. Top-20 chapters by FOB — latest complete year

# %%
cm_ref = mart_cm[mart_cm["CO_ANO"] == REF_YEAR]

chap_ref = (
    cm_ref.groupby("CO_CAPITULO")[["KG_LIQUIDO", "VL_FOB", "N_OPS"]]
    .sum()
    .sort_values("VL_FOB", ascending=False)
    .reset_index()
)

chap_ref["VL_FOB_bn"] = (chap_ref["VL_FOB"] / 1e9).round(2)
chap_ref["KG_MMt"]    = (chap_ref["KG_LIQUIDO"] / 1e6).round(1)  # thousand metric tons
chap_ref["share_pct"] = (chap_ref["VL_FOB"] / chap_ref["VL_FOB"].sum() * 100).round(1)
chap_ref["is_focus"]  = chap_ref["CO_CAPITULO"].isin(FOCUS_CHAPTERS)

top20 = chap_ref.head(20)

print(f"Top-20 chapters by FOB USD ({REF_YEAR}, full year):")
display(
    top20[["CO_CAPITULO", "VL_FOB_bn", "share_pct", "KG_MMt", "N_OPS"]]
    .rename(columns={"CO_CAPITULO": "Chapter", "VL_FOB_bn": "FOB USD bn",
                      "share_pct": "Share %", "KG_MMt": "kMt", "N_OPS": "Ops"})
    .reset_index(drop=True)
)

# Focus chapters in top-20?
focus_in_top20 = top20[top20["is_focus"]]["CO_CAPITULO"].tolist()
missing_focus  = [c for c in FOCUS_CHAPTERS if c not in focus_in_top20]
print(f"\nFocus chapters in top-20: {focus_in_top20}")
if missing_focus:
    print(f"WARNING: focus chapters NOT in top-20: {missing_focus}")
else:
    print("OK: all 5 focus chapters present in top-20.")

# %%
fig, ax = plt.subplots(figsize=(14, 6))

colors = ["#e04b3a" if c in FOCUS_CHAPTERS else "#4a90d9" for c in top20["CO_CAPITULO"]]
bars = ax.barh(top20["CO_CAPITULO"][::-1], top20["VL_FOB_bn"][::-1], color=colors[::-1])

ax.set_xlabel("FOB USD (billions)")
ax.set_title(f"Top-20 HS Chapters by Import FOB Value — Brazil {REF_YEAR} (full year)", pad=12)

# Label bars
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}B", va="center", fontsize=8)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#e04b3a", label="Focus chapters (28/29/30/38/39)"),
    Patch(color="#4a90d9", label="Other chapters"),
], loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(CHART_DIR / "market_top20_chapters.png", bbox_inches="tight")
plt.show()
print("Saved: market_top20_chapters.png")

# %% [markdown]
# ## 3. Focus chapters deep-dive — absolute size and share

# %%
focus_summary = (
    chap_ref[chap_ref["is_focus"]]
    .sort_values("VL_FOB", ascending=False)
    .reset_index(drop=True)
)

# Add rank within all chapters
rank_map = {r["CO_CAPITULO"]: i+1 for i, r in chap_ref.iterrows()}
focus_summary["rank_all"] = focus_summary["CO_CAPITULO"].map(rank_map)
focus_summary["label"] = focus_summary["CO_CAPITULO"].map(FOCUS_LABELS)

print(f"Focus chapter summary — {REF_YEAR} full year:")
display(
    focus_summary[["CO_CAPITULO", "label", "rank_all", "VL_FOB_bn", "share_pct", "N_OPS"]]
    .rename(columns={
        "CO_CAPITULO": "Chapter", "label": "Description",
        "rank_all": "Rank", "VL_FOB_bn": "FOB USD bn",
        "share_pct": "Share %", "N_OPS": "Ops",
    })
)

total_focus_fob = focus_summary["VL_FOB"].sum()
total_fob = chap_ref["VL_FOB"].sum()
print(f"\nCombined FOB USD (focus chapters): ${total_focus_fob/1e9:.1f}B "
      f"({100*total_focus_fob/total_fob:.1f}% of total market)")

# %% [markdown]
# ## 4. Origin concentration — HHI and CR4/CR10 by chapter

# %%
# HHI and concentration ratios per chapter using mart_ncm4_country
# Aggregate by chapter (CO_CAPITULO) + country
nc_with_chap = mart_nc.copy()
nc_with_chap["CO_CAPITULO"] = nc_with_chap["CO_POSICAO"].str[:2]

chap_country = (
    nc_with_chap.groupby(["CO_CAPITULO", "CO_PAIS", "NO_PAIS_ING"])["VL_FOB"]
    .sum()
    .reset_index()
)

def concentration_metrics(group: pd.DataFrame) -> pd.Series:
    """HHI (0–10000) and CR4/CR10 for a group already sorted by VL_FOB desc."""
    sorted_grp = group.sort_values("VL_FOB", ascending=False)
    total = sorted_grp["VL_FOB"].sum()
    if total == 0:
        return pd.Series({"HHI": np.nan, "CR4": np.nan, "CR10": np.nan, "n_origins": 0})
    shares = sorted_grp["VL_FOB"] / total
    hhi = round((shares ** 2).sum() * 10000, 0)
    cr4  = round(shares.head(4).sum() * 100, 1)
    cr10 = round(shares.head(10).sum() * 100, 1)
    return pd.Series({"HHI": hhi, "CR4": cr4, "CR10": cr10, "n_origins": len(sorted_grp)})

conc = (
    chap_country.groupby("CO_CAPITULO")
    .apply(concentration_metrics, include_groups=False)
    .reset_index()
)

# Merge with FOB totals for context
conc = conc.merge(
    chap_ref[["CO_CAPITULO", "VL_FOB_bn", "share_pct"]],
    on="CO_CAPITULO", how="left"
).sort_values("VL_FOB_bn", ascending=False)

conc["HHI_label"] = pd.cut(
    conc["HHI"],
    bins=[0, 1500, 2500, 10001],
    labels=["Competitive (<1500)", "Moderate (1500–2500)", "Concentrated (>2500)"],
)

print(f"Concentration metrics by chapter (top-30 by FOB, {REF_YEAR}):")
display(
    conc.head(30)[["CO_CAPITULO", "VL_FOB_bn", "n_origins", "CR4", "CR10", "HHI", "HHI_label"]]
    .rename(columns={"CO_CAPITULO": "Chapter", "VL_FOB_bn": "FOB bn",
                      "n_origins": "# Origins", "HHI_label": "Competition"})
    .reset_index(drop=True)
)

# %%
# Scatter: FOB scale vs. HHI — bubble = n_origins
focus_conc = conc[conc["CO_CAPITULO"].isin(FOCUS_CHAPTERS)]
other_conc = conc[~conc["CO_CAPITULO"].isin(FOCUS_CHAPTERS)].head(25)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(
    other_conc["VL_FOB_bn"], other_conc["HHI"],
    s=other_conc["n_origins"] * 3, alpha=0.5, color="#4a90d9", label="Other chapters"
)
ax.scatter(
    focus_conc["VL_FOB_bn"], focus_conc["HHI"],
    s=focus_conc["n_origins"] * 3, alpha=0.85, color="#e04b3a", zorder=5, label="Focus chapters"
)

for _, row in focus_conc.iterrows():
    ax.annotate(
        FOCUS_LABELS.get(row["CO_CAPITULO"], row["CO_CAPITULO"]),
        (row["VL_FOB_bn"], row["HHI"]),
        textcoords="offset points", xytext=(8, 4), fontsize=8,
    )

ax.axhline(1500, color="gray", linestyle="--", linewidth=0.8, label="HHI=1500 (competitive threshold)")
ax.axhline(2500, color="orange", linestyle="--", linewidth=0.8, label="HHI=2500 (concentrated threshold)")
ax.set_xlabel(f"FOB USD (billions) — {REF_YEAR}")
ax.set_ylabel("HHI (origin concentration)")
ax.set_title("Market size vs. origin concentration by HS Chapter — bubble = # of source countries")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CHART_DIR / "market_hhi_scatter.png", bbox_inches="tight")
plt.show()
print("Saved: market_hhi_scatter.png")

# %%
# Top-5 origins per focus chapter
for chap in FOCUS_CHAPTERS:
    grp = (
        chap_country[chap_country["CO_CAPITULO"] == chap]
        .sort_values("VL_FOB", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    if grp.empty:
        continue
    grp["share_pct"] = (grp["VL_FOB"] / chap_country[chap_country["CO_CAPITULO"] == chap]["VL_FOB"].sum() * 100).round(1)
    grp["VL_FOB_mm"] = (grp["VL_FOB"] / 1e6).round(1)
    print(f"\n{'='*55}")
    print(f"Chapter {chap} — {FOCUS_LABELS[chap]} — top-5 origins:")
    display(grp[["CO_PAIS", "NO_PAIS_ING", "VL_FOB_mm", "share_pct"]]
            .rename(columns={"CO_PAIS": "Code", "NO_PAIS_ING": "Country",
                              "VL_FOB_mm": "FOB USD MM", "share_pct": "Share %"}))

# %% [markdown]
# ## 5. YoY comparison — latest complete year vs. partial year annualized

# %%
if PARTIAL_YEAR is None:
    print("No partial year newer than the reference year — skipping annualized YoY.")
else:
    annualize = 12 / PARTIAL_MONTHS
    print(f"Annualization factor: 12 / {PARTIAL_MONTHS} = {annualize:.2f}x")
    print("WARNING: annualized figures are ESTIMATES. Seasonal effects not adjusted.")

# %%
if PARTIAL_YEAR is not None:
    cm_partial = mart_cm[mart_cm["CO_ANO"] == PARTIAL_YEAR]

    chap_partial = cm_partial.groupby("CO_CAPITULO")["VL_FOB"].sum().reset_index(name="VL_FOB_partial")
    chap_partial["VL_FOB_partial_ann"] = chap_partial["VL_FOB_partial"] * annualize

    yoy = chap_ref[["CO_CAPITULO", "VL_FOB"]].rename(columns={"VL_FOB": "VL_FOB_ref"})
    yoy = yoy.merge(chap_partial[["CO_CAPITULO", "VL_FOB_partial_ann"]], on="CO_CAPITULO", how="left")
    yoy["YoY_pct"] = ((yoy["VL_FOB_partial_ann"] / yoy["VL_FOB_ref"]) - 1) * 100
    yoy = yoy.sort_values("VL_FOB_ref", ascending=False)

    yoy["FOB_ref_bn"] = (yoy["VL_FOB_ref"] / 1e9).round(2)
    yoy["FOB_ann_bn"] = (yoy["VL_FOB_partial_ann"] / 1e9).round(2)
    yoy["is_focus"]   = yoy["CO_CAPITULO"].isin(FOCUS_CHAPTERS)

    print(f"YoY growth — top-20 chapters by {REF_YEAR} FOB ({PARTIAL_YEAR} annualized estimate):")
    display(
        yoy.head(20)[["CO_CAPITULO", "FOB_ref_bn", "FOB_ann_bn", "YoY_pct"]]
        .rename(columns={"CO_CAPITULO": "Chapter", "FOB_ref_bn": f"{REF_YEAR} FOB bn",
                          "FOB_ann_bn": f"{PARTIAL_YEAR} Ann. bn", "YoY_pct": "YoY %"})
        .round({"YoY %": 1})
        .reset_index(drop=True)
    )
    print(f"\nWARNING: {PARTIAL_YEAR} figures are annualized from {PARTIAL_MONTHS} months. "
          "Treat as directional, not precise.")

# %%
if PARTIAL_YEAR is not None and len(yoy) > 0:
    # YoY bar chart for focus chapters
    focus_yoy = yoy[yoy["is_focus"]].sort_values("YoY_pct", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: FOB absolute comparison (complete year vs. annualized partial)
    ax = axes[0]
    x = np.arange(len(focus_yoy))
    w = 0.35
    ax.bar(x - w/2, focus_yoy["FOB_ref_bn"], w, label=str(REF_YEAR), color="#4a90d9")
    ax.bar(x + w/2, focus_yoy["FOB_ann_bn"], w, label=f"{PARTIAL_YEAR} (ann.)",
           color="#e04b3a", alpha=0.85, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FOCUS_LABELS.get(c, c) for c in focus_yoy["CO_CAPITULO"]],
        rotation=15, ha="right", fontsize=8
    )
    ax.set_ylabel("FOB USD (billions)")
    ax.set_title(f"Focus chapters: {REF_YEAR} vs. {PARTIAL_YEAR} ann.")
    ax.legend(fontsize=8)

    # Right: YoY % for top-20 chapters
    ax2 = axes[1]
    yoy_top20 = yoy.head(20).sort_values("YoY_pct")
    colors_yoy = ["#e04b3a" if c in FOCUS_CHAPTERS else
                  ("#27ae60" if v >= 0 else "#c0392b")
                  for c, v in zip(yoy_top20["CO_CAPITULO"], yoy_top20["YoY_pct"])]
    ax2.barh(yoy_top20["CO_CAPITULO"], yoy_top20["YoY_pct"], color=colors_yoy)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel(f"YoY % ({PARTIAL_YEAR} ann. vs {REF_YEAR})")
    ax2.set_title("YoY growth — top-20 chapters")

    plt.suptitle(f"{PARTIAL_YEAR} annualized from {PARTIAL_MONTHS} months — directional only",
                 fontsize=9, color="darkorange", y=1.01)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "market_yoy_chapters.png", bbox_inches="tight")
    plt.show()
    print("Saved: market_yoy_chapters.png")

# %% [markdown]
# ## 6. Monthly trend — focus chapters (latest complete year)

# %%
trend = (
    mart_cm[
        (mart_cm["CO_ANO"] == REF_YEAR) &
        (mart_cm["CO_CAPITULO"].isin(FOCUS_CHAPTERS))
    ]
    .sort_values(["CO_CAPITULO", "CO_MES"])
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (metric, label, scale) in enumerate([
    ("VL_FOB", "FOB USD (millions)", 1e6),
    ("N_OPS",  "Operations",          1),
]):
    ax = axes[ax_idx]
    for chap, grp in trend.groupby("CO_CAPITULO"):
        ax.plot(grp["CO_MES"], grp[metric] / scale,
                marker="o", markersize=4,
                label=FOCUS_LABELS.get(chap, chap))
    ax.set_xlabel(f"Month ({REF_YEAR})")
    ax.set_ylabel(label)
    ax.set_title(f"{label} — focus chapters {REF_YEAR}")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

plt.tight_layout()
plt.savefig(CHART_DIR / "market_focus_monthly_trend.png", bbox_inches="tight")
plt.show()
print("Saved: market_focus_monthly_trend.png")

# %% [markdown]
# ## 7. Top-20 NCM-4 headings by FOB — latest complete year

# %%
ncm4_ref = (
    enriched[enriched["CO_ANO"] == REF_YEAR]
    .groupby("CO_POSICAO")[["KG_LIQUIDO", "VL_FOB"]]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
)
ncm4_ref["VL_FOB_mm"] = (ncm4_ref["VL_FOB"] / 1e6).round(1)
ncm4_ref["share_pct"] = (ncm4_ref["VL_FOB"] / ncm4_ref["VL_FOB"].sum() * 100).round(2)
ncm4_ref["is_focus"]  = ncm4_ref["CO_POSICAO"].str[:2].isin(FOCUS_CHAPTERS)

# Attach description if available
if "NO_NCM_POR" in enriched.columns:
    desc = (
        enriched[["CO_POSICAO", "NO_NCM_POR"]]
        .dropna(subset=["NO_NCM_POR"])
        .drop_duplicates("CO_POSICAO")
    )
    ncm4_ref = ncm4_ref.merge(desc, on="CO_POSICAO", how="left")
    display_cols = ["CO_POSICAO", "NO_NCM_POR", "VL_FOB_mm", "share_pct"]
else:
    display_cols = ["CO_POSICAO", "VL_FOB_mm", "share_pct"]

top20_ncm4 = ncm4_ref.head(20).reset_index(drop=True)
print(f"Top-20 NCM-4 headings by FOB USD ({REF_YEAR}):")
display(top20_ncm4[display_cols])

# %%
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["#e04b3a" if c in FOCUS_CHAPTERS else "#4a90d9"
          for c in top20_ncm4["CO_POSICAO"].str[:2]]
ax.barh(top20_ncm4["CO_POSICAO"][::-1], top20_ncm4["VL_FOB_mm"][::-1],
        color=colors[::-1])
ax.set_xlabel("FOB USD (millions)")
ax.set_title(f"Top-20 NCM-4 headings by FOB — Brazil {REF_YEAR} (red = focus chapters)")
plt.tight_layout()
plt.savefig(CHART_DIR / "market_top20_ncm4.png", bbox_inches="tight")
plt.show()
print("Saved: market_top20_ncm4.png")

# %% [markdown]
# ## 8. Port of entry (URF) — where does the volume land?

# %%
enr_ref = enriched[enriched["CO_ANO"] == REF_YEAR]
urf_summary = (
    enr_ref.groupby(["CO_URF", "NO_URF"])["VL_FOB"]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .head(15)
)
urf_summary["VL_FOB_bn"] = (urf_summary["VL_FOB"] / 1e9).round(2)
urf_summary["share_pct"] = (urf_summary["VL_FOB"] / enr_ref["VL_FOB"].sum() * 100).round(1)
urf_summary["label"] = urf_summary["NO_URF"].fillna(urf_summary["CO_URF"])

print(f"Top-15 ports of entry by FOB USD ({REF_YEAR}):")
display(urf_summary[["CO_URF", "label", "VL_FOB_bn", "share_pct"]]
        .rename(columns={"CO_URF": "URF Code", "label": "Port/URF",
                          "VL_FOB_bn": "FOB bn", "share_pct": "Share %"})
        .reset_index(drop=True))

# %%
# Focus chapters: which URFs handle them?
focus_urf = (
    enr_ref[enr_ref["CO_CAPITULO"].isin(FOCUS_CHAPTERS)]
    .groupby(["CO_URF", "NO_URF", "CO_CAPITULO"])["VL_FOB"]
    .sum()
    .reset_index()
)

# Pivot: URF × chapter heatmap-ready table
top_urfs = (
    focus_urf.groupby("CO_URF")["VL_FOB"].sum()
    .sort_values(ascending=False)
    .head(12).index
)

pivot_urf = (
    focus_urf[focus_urf["CO_URF"].isin(top_urfs)]
    .pivot_table(index="CO_URF", columns="CO_CAPITULO", values="VL_FOB",
                 aggfunc="sum", fill_value=0)
    / 1e6
).round(1)

# Replace URF codes with names where available
urf_name_map = (
    focus_urf[["CO_URF", "NO_URF"]].dropna(subset=["NO_URF"])
    .drop_duplicates("CO_URF")
    .set_index("CO_URF")["NO_URF"]
    .to_dict()
)
pivot_urf.index = [urf_name_map.get(c, c) for c in pivot_urf.index]
pivot_urf.columns.name = "Chapter"

print(f"FOB USD (millions) by port × focus chapter — {REF_YEAR}:")
display(pivot_urf.sort_values(pivot_urf.columns.tolist(), ascending=False))

# %% [markdown]
# ## 9. "So what?" — Sales intelligence summary

# %%
print("=" * 65)
print("  MARKET SIZING — SALES INTELLIGENCE SUMMARY")
print("=" * 65)

# 1. Focus chapter combined weight
total_fob_all = chap_ref["VL_FOB"].sum()
focus_fob = chap_ref[chap_ref["is_focus"]]["VL_FOB"].sum()
print(f"\nFocus chapters (28/29/30/38/39) combined {REF_YEAR} FOB:")
print(f"  ${focus_fob/1e9:.1f}B — {100*focus_fob/total_fob_all:.1f}% of total market")

# 2. Growth signal
if PARTIAL_YEAR is not None:
    focus_yoy_df = yoy[yoy["is_focus"]].copy()
    growing = focus_yoy_df[focus_yoy_df["YoY_pct"] > 0].sort_values("YoY_pct", ascending=False)
    if len(growing):
        print(f"\nGrowing focus chapters ({PARTIAL_YEAR} annualized vs {REF_YEAR}):")
        for _, r in growing.iterrows():
            print(f"  Ch{r['CO_CAPITULO']} {FOCUS_LABELS.get(r['CO_CAPITULO'],'')}: "
                  f"+{r['YoY_pct']:.1f}%")

# 3. Port concentration
top3_urf = urf_summary.head(3)
top3_share = top3_urf["share_pct"].sum()
top3_names = " / ".join(top3_urf["label"].tolist())
print(f"\nTop-3 ports handle {top3_share:.0f}% of total FOB:")
print(f"  {top3_names}")

print("\n" + "-" * 65)
print("QEntrega (air freight — GRU/VCP):")
print("  -> Ch30 (Pharma) and Ch29 (Org. chem): time-sensitive, air-eligible.")
print("  -> High unit value = freight cost is a small % of FOB = low price resistance.")
print("  -> Target: importers in SP/MG from US/EU origin at GRU/VCP.")
print()
print("Itatibense (road/sea — Santos, Itajai, Paranagua):")
print("  -> Ch39 (Plastics) and Ch28/38 (industrial chem): high volume, sea dominant.")
print("  -> Destination states: SP, SC, PR, MG.")
print("  -> Target: bulk chemical and plastic importers clearing through Santos/Itajai.")
print()
print("Data gap: no importer names in public Comex Stat.")
print("To unlock company-level prospecting: add Logcomex or ImportGenius DI data.")
