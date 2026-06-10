# %% [markdown]
# # 03 — Trade Lanes
#
# **Purpose:** Map origin → entry port (URF) → destination state (UF) flows.
# **Inputs:** `outputs/data/mart_tradeline.parquet`, `outputs/data/mart_sh6_via.parquet`
# **Output:** Lane rankings, heatmaps, modal split, focus-chapter analysis.
#
# > **Period note:** `mart_tradeline` covers all loaded years combined. YoY trend analysis
# > lives in `02_market_sizing.py`. Partial years are NOT annualised here.

# %%
from pathlib import Path

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

plt.rcParams.update({"figure.dpi": 110, "figure.figsize": (12, 5)})

try:
    display  # noqa: B018 — provided by IPython in interactive mode
except NameError:
    display = print

# %% [markdown]
# ## 1. Load data

# %%
FOCUS_CHAPTERS = ["28", "29", "30", "38", "39"]
FOCUS_LABELS   = {
    "28": "Ch28 Inorg Chem",
    "29": "Ch29 Org Chem",
    "30": "Ch30 Pharma",
    "38": "Ch38 Misc Chem",
    "39": "Ch39 Plastics",
}

# CO_VIA is zero-padded in the data ('01', '04', ...); unpadded variants kept defensively
VIA_GROUP = {
    "1": "Sea", "01": "Sea",
    "2": "River", "02": "River",
    "3": "Lake", "03": "Lake",
    "4": "Air", "04": "Air",
    "5": "Postal", "05": "Postal",
    "6": "Rail", "06": "Rail",
    "7": "Road", "07": "Road",
    "8": "Pipeline", "08": "Pipeline",
}
MODE_COLORS = {
    "Sea":      "#4393c3",
    "Air":      "#d73027",
    "Road":     "#1a9850",
    "River":    "#74add1",
    "Lake":     "#abd9e9",
    "Rail":     "#f46d43",
    "Postal":   "#fdae61",
    "Pipeline": "#878787",
    "Other":    "#cccccc",
}
MODES_ORDER = ["Sea", "Air", "Road", "Rail", "River", "Postal", "Pipeline", "Lake", "Other"]

# %%
lanes   = pd.read_parquet(DATA_DIR / "mart_tradeline.parquet")
sh6_via = pd.read_parquet(DATA_DIR / "mart_sh6_via.parquet")

# Period label for chart titles, derived from the monthly mart (tradeline has no year column)
_years = sorted(pd.read_parquet(DATA_DIR / "mart_chapter_month.parquet", columns=["CO_ANO"])["CO_ANO"].unique())
PERIOD_LABEL = f"{_years[0]}–{_years[-1]}" if len(_years) > 1 else str(_years[0])

# Resolve China code from data — no hardcoding
_cn = lanes[lanes["NO_PAIS_ING"].str.contains("China", case=False, na=False)]
CHINA_CODE = _cn["CO_PAIS"].iloc[0] if len(_cn) else "160"

print(f"mart_tradeline : {len(lanes):,} rows, columns: {list(lanes.columns)}")
print(f"mart_sh6_via   : {len(sh6_via):,} rows, columns: {list(sh6_via.columns)}")
print(f"Period         : {PERIOD_LABEL}")
print(f"China CO_PAIS resolved: {CHINA_CODE}")

# %%
# Null-safe labels
lanes["CO_PAIS"]  = lanes["CO_PAIS"].astype(str)
lanes["CO_URF"]   = lanes["CO_URF"].astype(str)
lanes["PAIS_LABEL"] = lanes["NO_PAIS_ING"].fillna("PAIS-" + lanes["CO_PAIS"])
lanes["URF_LABEL"]  = lanes["NO_URF"].fillna("URF-" + lanes["CO_URF"])

# Derive chapter from HS6 and map mode groups for mart_sh6_via
sh6_via["CO_CAPITULO"] = sh6_via["CO_SH6"].astype(str).str[:2]
sh6_via["CO_VIA"]      = sh6_via["CO_VIA"].astype(str)
sh6_via["MODE_GROUP"]  = sh6_via["CO_VIA"].map(VIA_GROUP).fillna("Other")

total_fob = lanes["VL_FOB"].sum()
print(f"Total FOB (combined period): USD {total_fob/1e9:.2f}B")

# %% [markdown]
# ## 2. Top-20 origin countries

# %%
origin_fob = (
    lanes.groupby(["CO_PAIS", "PAIS_LABEL"])["VL_FOB"]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .head(20)
)
origin_fob["VL_FOB_bn"]  = (origin_fob["VL_FOB"] / 1e9).round(2)
origin_fob["share_pct"]  = (origin_fob["VL_FOB"] / total_fob * 100).round(1)

print("Top-20 origin countries by FOB USD:")
display(origin_fob[["PAIS_LABEL", "VL_FOB_bn", "share_pct"]].reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#d73027" if c == CHINA_CODE else "#4393c3" for c in origin_fob["CO_PAIS"]]
ax.barh(origin_fob["PAIS_LABEL"][::-1], origin_fob["VL_FOB_bn"][::-1], color=colors[::-1])
ax.set_xlabel("FOB USD (billions)")
ax.set_title(f"Top-20 Origin Countries — FOB USD ({PERIOD_LABEL} combined)\nChina highlighted in red")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_top_origins.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Top-15 entry ports (URF)

# %%
urf_fob = (
    lanes.groupby(["CO_URF", "URF_LABEL"])["VL_FOB"]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .head(15)
)
urf_fob["VL_FOB_bn"] = (urf_fob["VL_FOB"] / 1e9).round(2)
urf_fob["share_pct"] = (urf_fob["VL_FOB"] / total_fob * 100).round(1)

print("Top-15 entry ports (URF) by FOB USD:")
display(urf_fob[["URF_LABEL", "VL_FOB_bn", "share_pct"]].reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(urf_fob["URF_LABEL"][::-1], urf_fob["VL_FOB_bn"][::-1], color="#4393c3")
ax.set_xlabel("FOB USD (billions)")
ax.set_title(f"Top-15 Entry Ports (URF) — FOB USD ({PERIOD_LABEL} combined)")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_top_urfs.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Top-15 destination states (SG_UF_NCM)

# %%
uf_fob = (
    lanes.groupby("SG_UF_NCM")["VL_FOB"]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .head(15)
)
uf_fob["VL_FOB_bn"] = (uf_fob["VL_FOB"] / 1e9).round(2)
uf_fob["share_pct"] = (uf_fob["VL_FOB"] / total_fob * 100).round(1)

print("Top-15 destination states by FOB USD:")
display(uf_fob[["SG_UF_NCM", "VL_FOB_bn", "share_pct"]].reset_index(drop=True))

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(uf_fob["SG_UF_NCM"][::-1], uf_fob["VL_FOB_bn"][::-1], color="#74add1")
ax.set_xlabel("FOB USD (billions)")
ax.set_title(f"Top-15 Destination States — FOB USD ({PERIOD_LABEL} combined)")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fB"))
plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_top_states.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Origin × URF heatmap
#
# Row-normalised: each cell shows what % of that country's total FOB enters via each port.
# Restricted to top-15 origins and top-12 URFs for readability.

# %%
top_origins = origin_fob["PAIS_LABEL"].tolist()[:15]
top_urfs    = urf_fob["URF_LABEL"].tolist()[:12]

sub_ou = (
    lanes[lanes["PAIS_LABEL"].isin(top_origins) & lanes["URF_LABEL"].isin(top_urfs)]
    .groupby(["PAIS_LABEL", "URF_LABEL"])["VL_FOB"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=top_origins, columns=top_urfs, fill_value=0)
)

row_tot = sub_ou.sum(axis=1).replace(0, np.nan)
sub_ou_pct = sub_ou.div(row_tot, axis=0) * 100
sub_ou_pct_disp = sub_ou_pct.where(sub_ou_pct > 0)  # 0% -> NaN so it renders as grey

cmap_ou = plt.cm.YlOrRd.copy()
cmap_ou.set_bad("#e0e0e0")

fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(sub_ou_pct_disp.values, aspect="auto", cmap=cmap_ou, vmin=0, vmax=80)
plt.colorbar(im, ax=ax, label="% of origin's FOB")

ax.set_xticks(range(len(top_urfs)))
ax.set_xticklabels(top_urfs, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(top_origins)))
ax.set_yticklabels(top_origins, fontsize=8)
ax.set_title(f"Origin × URF — % of Origin's FOB through each Port ({PERIOD_LABEL})", pad=12)

for i in range(len(top_origins)):
    for j in range(len(top_urfs)):
        val = sub_ou_pct.values[i, j]
        if val >= 5:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color="black" if val < 55 else "white")

plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_origin_urf_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. URF × State heatmap
#
# Row-normalised: each cell shows what % of that port's FOB clears to each destination state.

# %%
top_states = uf_fob["SG_UF_NCM"].tolist()[:12]

sub_us = (
    lanes[lanes["URF_LABEL"].isin(top_urfs) & lanes["SG_UF_NCM"].isin(top_states)]
    .groupby(["URF_LABEL", "SG_UF_NCM"])["VL_FOB"]
    .sum()
    .unstack(fill_value=0)
    .reindex(index=top_urfs, columns=top_states, fill_value=0)
)

row_tot_us = sub_us.sum(axis=1).replace(0, np.nan)
sub_us_pct = sub_us.div(row_tot_us, axis=0) * 100
sub_us_pct_disp = sub_us_pct.where(sub_us_pct > 0)  # 0% -> NaN so it renders as grey

cmap_us = plt.cm.PuBuGn.copy()
cmap_us.set_bad("#e0e0e0")

fig, ax = plt.subplots(figsize=(12, 7))
im2 = ax.imshow(sub_us_pct_disp.values, aspect="auto", cmap=cmap_us, vmin=0, vmax=80)
plt.colorbar(im2, ax=ax, label="% of port's FOB")

ax.set_xticks(range(len(top_states)))
ax.set_xticklabels(top_states, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(top_urfs)))
ax.set_yticklabels(top_urfs, fontsize=8)
ax.set_title(f"URF × Destination State — % of Port's FOB to each State ({PERIOD_LABEL})", pad=12)

for i in range(len(top_urfs)):
    for j in range(len(top_states)):
        val = sub_us_pct.values[i, j]
        if val >= 5:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="black" if val < 55 else "white")

plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_urf_state_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Top-20 complete trade lanes
#
# Full lane = Origin country → URF → Destination state.

# %%
lane_full = (
    lanes.groupby(["PAIS_LABEL", "URF_LABEL", "SG_UF_NCM"])
    .agg(VL_FOB=("VL_FOB", "sum"), KG_LIQUIDO=("KG_LIQUIDO", "sum"), N_OPS=("N_OPS", "sum"))
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .head(20)
)
lane_full["VL_FOB_mn"]  = (lane_full["VL_FOB"] / 1e6).round(1)
lane_full["KG_000t"]    = (lane_full["KG_LIQUIDO"] / 1e6).round(1)
lane_full["share_pct"]  = (lane_full["VL_FOB"] / total_fob * 100).round(2)

print("Top-20 complete trade lanes (Origin -> URF -> State):")
display(lane_full[["PAIS_LABEL", "URF_LABEL", "SG_UF_NCM",
                   "VL_FOB_mn", "KG_000t", "N_OPS", "share_pct"]].reset_index(drop=True))

# %% [markdown]
# ## 8. Modal split by chapter
#
# Source: `mart_sh6_via` (HS6 × transport mode). Chapter derived as `CO_SH6[:2]`.
# Top-20 chapters by total FOB; Air% sort highlights pharma/electronics.

# %%
modal = (
    sh6_via.groupby(["CO_CAPITULO", "MODE_GROUP"])["VL_FOB"]
    .sum()
    .reset_index()
)

# Top-20 chapters by total FOB
chapter_totals = modal.groupby("CO_CAPITULO")["VL_FOB"].sum().nlargest(20)
modal_top = modal[modal["CO_CAPITULO"].isin(chapter_totals.index)]

# Pivot to wide, row-normalise
pivot_modal = (
    modal_top.pivot_table(index="CO_CAPITULO", columns="MODE_GROUP", values="VL_FOB", fill_value=0)
)
pivot_modal_pct = pivot_modal.div(pivot_modal.sum(axis=1), axis=0) * 100

# Sort by Air%
if "Air" in pivot_modal_pct.columns:
    pivot_modal_pct = pivot_modal_pct.sort_values("Air", ascending=True)

# Add chapter labels
pivot_modal_pct.index = [
    FOCUS_LABELS.get(c, f"Ch{c}") for c in pivot_modal_pct.index
]

fig, ax = plt.subplots(figsize=(12, 8))
bottom = np.zeros(len(pivot_modal_pct))
for mode in MODES_ORDER:
    if mode in pivot_modal_pct.columns:
        vals = pivot_modal_pct[mode].values
        ax.barh(range(len(pivot_modal_pct)), vals, left=bottom,
                label=mode, color=MODE_COLORS[mode])
        bottom += vals

ax.set_yticks(range(len(pivot_modal_pct)))
ax.set_yticklabels(pivot_modal_pct.index, fontsize=8)
ax.set_xlabel("Share of FOB (%)")
ax.set_title(f"Modal Split by Chapter — FOB % ({PERIOD_LABEL}, top-20 chapters, sorted by Air%)")
ax.set_xlim(0, 100)
ax.legend(loc="lower right", fontsize=8)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())

# Highlight focus chapters
for i, label in enumerate(pivot_modal_pct.index):
    if any(v in label for v in FOCUS_LABELS.values()):
        ax.get_yticklabels()[i].set_fontweight("bold")

plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_modal_split_chapter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Focus chapter × URF flows
#
# For Chapters 28/29/30/38/39: which entry ports handle the most volume?
# China spotlight: same analysis restricted to China-origin cargo.

# %%
focus_lanes = lanes[lanes["CO_CAPITULO"].isin(FOCUS_CHAPTERS)].copy()
focus_lanes["CHAPTER_LABEL"] = focus_lanes["CO_CAPITULO"].map(FOCUS_LABELS)

focus_urf = (
    focus_lanes.groupby(["CHAPTER_LABEL", "URF_LABEL"])["VL_FOB"]
    .sum()
    .reset_index()
)

# Top-8 URFs per chapter
top_urf_per_chap = (
    focus_urf.sort_values(["CHAPTER_LABEL", "VL_FOB"], ascending=[True, False])
    .groupby("CHAPTER_LABEL")
    .head(8)
)

for chap_label, grp in top_urf_per_chap.groupby("CHAPTER_LABEL"):
    grp = grp.sort_values("VL_FOB", ascending=False)
    print(f"\n{chap_label} — top ports by FOB:")
    grp_disp = grp.copy()
    grp_disp["VL_FOB_mn"] = (grp_disp["VL_FOB"] / 1e6).round(1)
    display(grp_disp[["URF_LABEL", "VL_FOB_mn"]].reset_index(drop=True))

# %%
# Pivot: chapter rows × top-10 URFs columns, normalised by chapter
top10_urfs_focus = (
    focus_urf.groupby("URF_LABEL")["VL_FOB"]
    .sum()
    .nlargest(10)
    .index.tolist()
)

focus_pivot = (
    focus_urf[focus_urf["URF_LABEL"].isin(top10_urfs_focus)]
    .pivot_table(index="CHAPTER_LABEL", columns="URF_LABEL", values="VL_FOB", fill_value=0)
)
focus_pivot_pct = focus_pivot.div(focus_pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 4))
bottom = np.zeros(len(focus_pivot_pct))
colors_urf = plt.cm.tab10(np.linspace(0, 1, len(top10_urfs_focus)))
for i, urf in enumerate(top10_urfs_focus):
    if urf in focus_pivot_pct.columns:
        vals = focus_pivot_pct[urf].values
        ax.bar(range(len(focus_pivot_pct)), vals, bottom=bottom,
               label=urf, color=colors_urf[i])
        bottom += vals

ax.set_xticks(range(len(focus_pivot_pct)))
ax.set_xticklabels(focus_pivot_pct.index, rotation=20, ha="right")
ax.set_ylabel("Share of FOB (%)")
ax.set_title(f"Focus Chapters — Port Distribution (% of Chapter FOB, {PERIOD_LABEL})")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="upper right", fontsize=7, title="URF", title_fontsize=7)
plt.tight_layout()
plt.savefig(CHART_DIR / "lanes_focus_urf.png", bbox_inches="tight")
plt.show()

# %%
# China spotlight: focus chapters, China origin only
china_focus = focus_lanes[focus_lanes["CO_PAIS"] == CHINA_CODE]

if len(china_focus) == 0:
    print("No China rows in focus chapters — check CO_PAIS values.")
else:
    china_urf = (
        china_focus.groupby(["CHAPTER_LABEL", "URF_LABEL"])["VL_FOB"]
        .sum()
        .reset_index()
        .sort_values(["CHAPTER_LABEL", "VL_FOB"], ascending=[True, False])
    )
    print("China-origin focus chapters — top URFs by FOB:")
    display(
        china_urf.assign(VL_FOB_mn=lambda d: (d.VL_FOB/1e6).round(1))
        .groupby("CHAPTER_LABEL")
        .head(5)
        [["CHAPTER_LABEL", "URF_LABEL", "VL_FOB_mn"]]
        .reset_index(drop=True)
    )

# %% [markdown]
# ## 10. So What? — Strategic read-out
#
# Key signals for QEntrega and Itatibense Transportes.

# %%
print("=" * 60)
print("  STRATEGIC READ-OUT — TRADE LANE INTELLIGENCE")
print("=" * 60)

# China share
china_fob = lanes[lanes["CO_PAIS"] == CHINA_CODE]["VL_FOB"].sum()
print(f"\nChina share of total FOB: {100 * china_fob / total_fob:.1f}%")

# Top-3 URFs
print(f"\nTop-3 entry ports:")
for _, row in urf_fob.head(3).iterrows():
    print(f"  {row['URF_LABEL']:35s} USD {row['VL_FOB_bn']:.2f}B ({row['share_pct']:.1f}%)")

# Top-3 states
print(f"\nTop-3 destination states:")
for _, row in uf_fob.head(3).iterrows():
    print(f"  {row['SG_UF_NCM']:5s} USD {row['VL_FOB_bn']:.2f}B ({row['share_pct']:.1f}%)")

# Focus chapter URF concentration
print("\nFocus chapter dominant ports (top-1 URF per chapter):")
focus_urf_top1 = (
    focus_lanes.groupby(["CHAPTER_LABEL", "URF_LABEL"])["VL_FOB"]
    .sum()
    .reset_index()
    .sort_values("VL_FOB", ascending=False)
    .groupby("CHAPTER_LABEL")
    .first()
    .reset_index()
)
for _, row in focus_urf_top1.iterrows():
    print(f"  {row['CHAPTER_LABEL']:20s} -> {row['URF_LABEL']:35s} USD {row['VL_FOB']/1e6:.0f}M")

print("\n--- QEntrega (air, GRU/VCP, pharma/electronics) ---")
print("  Look for: chapters with >20% Air modal share arriving at Guarulhos/Viracopos")
print("  These segments = time-critical, temperature-controlled, high-margin opportunities")

print("\n--- Itatibense Transportes (sea/road, Santos/Itajai/Paranagua, plastics/chem) ---")
print("  Look for: Ch39 (plastics), Ch28/29/38 (chem) via Santos with SP/MG destination")
print("  Road leg from port to warehouse in SP interior = core Itatibense territory")

print("\n--- Data gap ---")
print("  Comex Stat has no importer names. These flows identify the segments.")
print("  Add Logcomex / ImportGenius DI data to name the actual importers per lane.")
