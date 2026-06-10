# Comex-stats — Brazilian Import Trade Intelligence

Trade intelligence analysis of Brazilian import data (Comex Stat / SISCOMEX extracts) for
**QEntrega** (air freight) and **Itatibense Transportes** (road/sea logistics): market sizing,
trade lanes, pricing benchmarks, freight intensity, and NCM opportunity scoring.

## Project layout

```
Config/config.xlsx          Analytical parameters (weights, thresholds, target URFs, seasonality)
Context/instructions.txt    Analytical methodology and standards
Data/Imports Data/          IMP_<YEAR>.csv raw Comex Stat extracts (gitignored — large)
Data/References/            NCM / PAIS / URF / VIA / UF lookup tables
pipeline/                   Staged DuckDB pipeline (run in order 00 → 03)
notebooks/                  Analysis notebooks (.py percent scripts, run in VS Code)
outputs/data/               Pipeline parquet artifacts (gitignored — regeneratable)
outputs/charts/             Chart PNGs (gitignored — regeneratable)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python pipeline/setup_config.py   # one-off: adds the 'target_urfs' sheet to Config/config.xlsx
```

> On Eli Lilly managed machines, public PyPI is blocked — configure pip/uv to use the
> internal Artifactory index (`elilillyco.jfrog.io`) with an identity token first.

## Running the pipeline

Drop one or more `IMP_<YEAR>.csv` files (Comex Stat format, `;`-separated, Latin-1) into
`Data/Imports Data/`, then run from the project root:

```bash
python pipeline/00_ingest.py    # parse CSVs, profile, reject-row report  -> raw_combined.parquet
python pipeline/01_clean.py     # NCM validation, outlier flags, periods  -> clean.parquet
python pipeline/02_enrich.py    # reference joins, CIF, freight metrics   -> enriched.parquet
python pipeline/03_marts.py     # pre-aggregated analytical marts         -> mart_*.parquet
```

The pipeline is fully dynamic — adding a new year requires no code changes. Years with 12
distinct months are detected as *complete*; the latest complete year drives all rankings,
and partial years are used only for clearly-labeled annualized estimates / momentum signals.

## Notebooks

`.py` files with `# %%` cell markers — open in VS Code and run cells interactively
(Python extension), or execute top-to-bottom as plain scripts.

| Notebook | Purpose |
|---|---|
| `01_data_profile.py` | Data quality sign-off (go/no-go before any analysis) |
| `02_market_sizing.py` | Market size by chapter/NCM-4, concentration, YoY |
| `03_trade_lanes.py` | Origin → port (URF) → destination state flows, modal split |
| `04_pricing_benchmarks.py` | Median/P10/P90 USD/kg by NCM-4 and origin, China price index |
| `05_freight_analysis.py` | Freight intensity (% of FOB) by chapter, mode, origin |
| `06_china_imports.py` | China deep-dive: HS6 segments, ports, modes, trends |
| `07_ncm_scoring.py` | Composite NCM-4 attractiveness ranking (6 weighted criteria) |
| `08_seasonality.py` | Intra-year seasonality by chapter |

## Configuration (Config/config.xlsx)

| Sheet | Used by | Contents |
|---|---|---|
| `outlier_detection` | pipeline/01_clean.py | `iqr_multiplier`, `min_ops_por_grupo` |
| `scoring_weights` | 07_ncm_scoring.py | Criterion weights (`criterio`, `peso`) — must sum to 1 |
| `scoring_thresholds` | 07_ncm_scoring.py | `top_n_ncms`, `vol_min_fob_usd`, `crescimento_cagr_anos`, HHI/freight thresholds |
| `target_urfs` | 07_ncm_scoring.py | Target ports for route scoring (`co_urf`, `descricao`) — 7-digit zero-padded codes |
| `seasonality` | 08_seasonality.py | `anos` (`;`-separated or `auto`), `capitulos_foco` |

## Data conventions

- All code columns (`CO_NCM`, `CO_PAIS`, `CO_URF`, `CO_VIA`, `CO_SH6`, ...) are **zero-padded
  strings**, never integers — e.g. air freight is `CO_VIA = '04'`, Porto de Santos is
  `CO_URF = '0817800'`.
- Unit-price statistics (median/P10/P90) exclude IQR-flagged outlier rows; volume/value sums
  always include all declared rows.
- `FREIGHT_PCT_FOB` is the preferred freight-intensity metric; `FREIGHT_PER_KG` is a proxy
  (net weight ≠ chargeable weight).
- Public Comex Stat is anonymized — no importer CNPJs/names. Company-level prospecting
  requires a Logcomex / ImportGenius / DI-level overlay.
