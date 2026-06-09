"""
Shared paths, file-discovery helpers, and dtype specs for all pipeline stages.
Analytical parameters (weights, thresholds, etc.) live in Config/config.xlsx.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "Data" / "Imports Data"
REFS_DIR     = PROJECT_ROOT / "Data" / "References"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "data"
CONFIG_XLSX  = PROJECT_ROOT / "Config" / "config.xlsx"


def discover_import_files() -> list[Path]:
    """Return sorted list of all IMP_*.csv files found in DATA_DIR."""
    files = sorted(DATA_DIR.glob("IMP_*.csv"))
    if not files:
        raise FileNotFoundError(f"No IMP_*.csv files found in {DATA_DIR}")
    return files


def import_glob_str() -> str:
    """Return a glob string DuckDB can use to read all import CSVs in one pass."""
    return str(DATA_DIR / "IMP_*.csv").replace("\\", "/")


def ref_path(filename: str) -> str:
    """Return forward-slash path string for a reference CSV (for DuckDB)."""
    return str(REFS_DIR / filename).replace("\\", "/")


# Column dtype groups shared across pipeline stages
STRING_COLS = ["CO_NCM", "CO_UNID", "CO_PAIS", "SG_UF_NCM", "CO_VIA", "CO_URF"]

NUMERIC_INT_COLS = ["CO_ANO", "CO_MES", "QT_ESTAT", "KG_LIQUIDO"]

NUMERIC_FLOAT_COLS = ["VL_FOB", "VL_FRETE", "VL_SEGURO"]
