"""
Stage 2 — Enrich

Input:  outputs/data/clean.parquet
Output: outputs/data/enriched.parquet

What it does:
  - Left-joins all reference tables (NCM, PAIS, URF, VIA, UF) via DuckDB SQL
  - Adds derived fields (VL_CIF, FREIGHT_PER_KG, FREIGHT_PCT_FOB, CO_CAPITULO, PERIODO)
  - Reports unmatched join rates and sample unmatched keys per reference table
  - FREIGHT_PCT_FOB is NULL when VL_FRETE is missing (not coerced to 0)

Run from project root:
    python pipeline/02_enrich.py
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, ref_path

INPUT  = OUTPUT_DIR / "clean.parquet"
OUTPUT = OUTPUT_DIR / "enriched.parquet"


def _csv(filename: str) -> str:
    """DuckDB read_csv expression for a reference CSV in Data/References/."""
    path = ref_path(filename)
    return f"read_csv('{path}', sep=';', quote='\"', encoding='Latin-1', ignore_errors=true)"


def main() -> None:
    print("=" * 60)
    print("  STAGE 2 — ENRICH")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run pipeline/01_clean.py first.")

    in_str  = str(INPUT).replace("\\", "/")
    out_str = str(OUTPUT).replace("\\", "/")
    con = duckdb.connect()

    # -------------------------------------------------------------------------
    # Register clean data and reference views
    # -------------------------------------------------------------------------
    con.execute(f"CREATE OR REPLACE VIEW clean AS SELECT * FROM read_parquet('{in_str}')")

    con.execute(f"""
        CREATE OR REPLACE VIEW ncm_ref AS
        SELECT CO_NCM, CO_SH6, NO_NCM_POR, NO_NCM_ING, CO_PPE, CO_FAT_AGREG
        FROM {_csv('NCM.csv')}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW pais_ref AS
        SELECT CO_PAIS, CO_PAIS_ISON3, CO_PAIS_ISOA3, NO_PAIS, NO_PAIS_ING
        FROM {_csv('PAIS.csv')}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW urf_ref AS
        SELECT CO_URF, NO_URF
        FROM {_csv('URF.csv')}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW via_ref AS
        SELECT CO_VIA, NO_VIA
        FROM {_csv('VIA.csv')}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW uf_ref AS
        SELECT SG_UF, NO_UF, NO_REGIAO
        FROM {_csv('UF.csv')}
    """)

    n_rows = con.execute("SELECT COUNT(*) FROM clean").fetchone()[0]
    print(f"\nReading {INPUT.name} ... {n_rows:,} rows")

    # -------------------------------------------------------------------------
    # Single JOIN + derived columns
    # -------------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE VIEW enriched AS
        SELECT
            -- All clean columns
            c.*,

            -- NCM reference
            n.CO_SH6,
            n.NO_NCM_POR,
            n.NO_NCM_ING,
            n.CO_PPE,
            n.CO_FAT_AGREG,

            -- PAIS reference
            p.CO_PAIS_ISON3,
            p.CO_PAIS_ISOA3,
            p.NO_PAIS,
            p.NO_PAIS_ING,

            -- URF reference
            u.NO_URF,

            -- VIA reference
            v.NO_VIA,

            -- UF reference
            f.NO_UF,
            f.NO_REGIAO,

            -- HS chapter (first 2 chars of NCM)
            LEFT(c.CO_NCM, 2) AS CO_CAPITULO,

            -- Period integer YYYYMM for easy sorting
            CAST(c.CO_ANO AS BIGINT) * 100 + CAST(c.CO_MES AS BIGINT) AS PERIODO,

            -- CIF (best-estimate: treat missing freight/insurance as 0)
            c.VL_FOB
                + COALESCE(c.VL_FRETE,  0)
                + COALESCE(c.VL_SEGURO, 0)                         AS VL_CIF,

            -- CIF unit price
            CASE
                WHEN c.KG_LIQUIDO > 0 THEN
                    (c.VL_FOB
                     + COALESCE(c.VL_FRETE,  0)
                     + COALESCE(c.VL_SEGURO, 0)) / c.KG_LIQUIDO
                ELSE NULL
            END                                                     AS UNIT_CIF_PER_KG,

            -- Freight per kg (proxy; only when freight is known and weight > 0)
            CASE
                WHEN c.VL_FRETE IS NOT NULL AND c.KG_LIQUIDO > 0
                THEN c.VL_FRETE / c.KG_LIQUIDO
                ELSE NULL
            END                                                     AS FREIGHT_PER_KG,

            -- Freight % of FOB (NULL when freight unknown — not coerced to 0)
            CASE
                WHEN c.VL_FRETE IS NOT NULL AND c.VL_FOB > 0
                THEN c.VL_FRETE / c.VL_FOB
                ELSE NULL
            END                                                     AS FREIGHT_PCT_FOB

        FROM clean c
        LEFT JOIN ncm_ref  n ON c.CO_NCM    = n.CO_NCM
        LEFT JOIN pais_ref p ON c.CO_PAIS   = p.CO_PAIS
        LEFT JOIN urf_ref  u ON c.CO_URF    = u.CO_URF
        LEFT JOIN via_ref  v ON c.CO_VIA    = v.CO_VIA
        LEFT JOIN uf_ref   f ON c.SG_UF_NCM = f.SG_UF
    """)

    # -------------------------------------------------------------------------
    # Report unmatched rates + sample keys
    # -------------------------------------------------------------------------
    print("\nJoin match rates (unmatched = no lookup entry found):")

    join_checks = [
        ("NCM",  "NO_NCM_ING",  "CO_NCM"),
        ("PAIS", "NO_PAIS_ING", "CO_PAIS"),
        ("URF",  "NO_URF",      "CO_URF"),
        ("VIA",  "NO_VIA",      "CO_VIA"),
        ("UF",   "NO_UF",       "SG_UF_NCM"),
    ]

    for ref_name, sentinel_col, key_col in join_checks:
        result = con.execute(f"""
            SELECT
                SUM(CASE WHEN {sentinel_col} IS NULL THEN 1 ELSE 0 END) AS n_unmatched,
                COUNT(*) AS n_total
            FROM enriched
        """).fetchone()
        n_unmatched, n_total = result
        pct = 100 * n_unmatched / max(n_total, 1)
        print(f"  {ref_name:6s}: {n_unmatched:,} unmatched ({pct:.1f}%)", end="")

        if n_unmatched > 0:
            samples = con.execute(f"""
                SELECT DISTINCT {key_col}
                FROM enriched
                WHERE {sentinel_col} IS NULL AND {key_col} IS NOT NULL
                LIMIT 5
            """).fetchall()
            sample_vals = [str(r[0]) for r in samples]
            print(f"  — sample keys: {sample_vals}", end="")
        print()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    n_cols = len(con.execute("SELECT * FROM enriched LIMIT 0").description)
    print(f"\nFinal schema: {n_rows:,} rows × {n_cols} columns")

    print(f"Writing {OUTPUT.name} ...")
    con.execute(f"""
        COPY (SELECT * FROM enriched)
        TO '{out_str}'
        (FORMAT PARQUET, CODEC 'SNAPPY')
    """)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"  Done. {size_mb:.1f} MB")
    print("\nStage 2 complete. Next: pipeline/03_marts.py")


if __name__ == "__main__":
    main()
