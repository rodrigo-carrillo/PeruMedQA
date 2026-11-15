#!/usr/bin/env python3
"""
consolidate_exams.py — Consolida todos los exámenes CONAREME (YearXXXX/<examen>.json) en dos variantes:

  1. conareme_master_5opciones          → todos los exámenes con 5 alternativas; en los exámenes que
                                          originalmente tenían 4 (2023–2026), option_E = "Ninguna de las anteriores".
  2. conareme_master_opciones_originales → cada examen conserva su número original de alternativas; en los
                                          exámenes de 4 opciones, option_E queda vacía (missing).

Cada variante se guarda en CSV (utf-8-sig, todas las celdas entre comillas), JSON y Parquet en ../00_Datasets/.
Se generan además un reporte de validación (JSON) y un log en logs/.

Fuente canónica: los JSON por examen generados por process_exams.py (se valida que coincidan con sus CSV).

Uso:
    python3 consolidate_exams.py
    python3 consolidate_exams.py --output-dir /otra/carpeta
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR.parent / "00_Datasets"
LOG_DIR = BASE_DIR / "logs"
FILLER_E = "Ninguna de las anteriores"
OPTION_COLS = [f"option_{l}" for l in "ABCDE"]
COLUMNS = ["question_id", "year", "exam", "exam_type", "exam_title", "specialty", "question_number",
           "n_options_original", "questions", *OPTION_COLS, "correct_answer", "has_image", "flags"]

# Normalización de especialidad a partir del título oficial del encabezado del PDF
# (clave: título en minúsculas y sin tildes).
SPECIALTY_MAP = {
    "patologia": "Anatomía Patológica",
    "anatomia patologica": "Anatomía Patológica",
    "anatomia patologica laboratorio y patologia": "Anatomía Patológica",
    "ginecologia y obstetricia": "Ginecología y Obstetricia",
    "ginecologia obstetricia": "Ginecología y Obstetricia",
    "ginecologia - obstetricia": "Ginecología y Obstetricia",
    "cirugia de torax y cardiovascular": "Cirugía de Tórax y Cardiovascular",
    "cirugia cardiovascular": "Cirugía Cardiovascular",
    "cirugia general": "Cirugía General",
    "anestesiologia": "Anestesiología",
    "neurocirugia": "Neurocirugía",
    "oftalmologia": "Oftalmología",
    "ortopedia y traumatologia": "Ortopedia y Traumatología",
    "pediatria": "Pediatría",
    "psiquiatria": "Psiquiatría",
    "radiologia": "Radiología",
    "urologia": "Urología",
    "prueba a": "Prueba A",
    "prueba b": "Prueba B",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", strip_accents(s).upper()).strip("_")


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def setup_logger(stamp: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("consolidacion")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_DIR / f"consolidacion_{stamp}.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


class Report:
    def __init__(self):
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.checks: list[str] = []

    def error(self, log, check, msg, **kw):
        self.errors.append({"check": check, "message": msg, **kw}); log.error(f"  ✗ {msg}")

    def warn(self, log, check, msg, **kw):
        self.warnings.append({"check": check, "message": msg, **kw}); log.warning(f"  ⚠ {msg}")


def load_exam(json_path: Path, rep: Report, log: logging.Logger) -> tuple[list[dict], dict]:
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    md = data["metadata"]
    year, exam = int(md["year"]), unicodedata.normalize("NFC", md["exam"])
    letters = md["options"]
    title = md.get("exam_title") or exam
    key = strip_accents(title).lower().strip()
    specialty = SPECIALTY_MAP.get(key)
    if specialty is None:
        specialty = title
        rep.warn(log, "specialty_mapping", f"{year} {exam}: título '{title}' sin mapeo de especialidad; se usa tal cual")
    rows = []
    for q in data["questions"]:
        row = {
            "question_id": f"{year}_{slug(exam)}_{int(q['number']):03d}",
            "year": year, "exam": exam, "exam_type": md.get("exam_type"), "exam_title": title,
            "specialty": specialty, "question_number": int(q["number"]), "n_options_original": len(letters),
            "questions": q["question"],
        }
        for l in "ABCDE":
            row[f"option_{l}"] = q["options"].get(l) if l in letters else None
        row["correct_answer"] = q["correct_answer"]
        row["has_image"] = bool(q.get("has_image"))
        row["flags"] = ";".join(q.get("flags") or [])
        rows.append(row)

    # Validar que el CSV por examen coincide con el JSON
    csv_path = json_path.with_suffix(".csv")
    info = {"year": year, "exam": exam, "json": str(json_path.relative_to(BASE_DIR)), "rows": len(rows),
            "n_options": len(letters), "csv_matches_json": None}
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        ok = len(df) == len(rows)
        if ok:
            for r, (_, c) in zip(rows, df.iterrows()):
                if (int(c["question_number"]) != r["question_number"] or c["questions"] != r["questions"]
                        or c["correct_answer"] != (r["correct_answer"] or "")
                        or any(c[f"option_{l}"] != r[f"option_{l}"] for l in letters)):
                    ok = False
                    break
        info["csv_matches_json"] = ok
        if not ok:
            rep.error(log, "csv_vs_json", f"{year} {exam}: el CSV no coincide con el JSON")
    else:
        rep.error(log, "csv_vs_json", f"{year} {exam}: falta el CSV")
    return rows, info


def validate(master: pd.DataFrame, files: list[dict], rep: Report, log: logging.Logger):
    log.info("[validación] expected_columns ...")
    rep.checks.append("expected_columns")
    if list(master.columns) != COLUMNS:
        rep.error(log, "expected_columns", f"columnas inesperadas: {list(master.columns)}")
    else:
        log.info("  ✓ Estructura correcta")

    log.info("[validación] row_counts_reconciliation ...")
    rep.checks.append("row_counts_reconciliation")
    counts = master.groupby(["year", "exam"]).size()
    bad = [f for f in files if counts.get((f["year"], f["exam"]), 0) != f["rows"]]
    if bad or len(master) != sum(f["rows"] for f in files):
        rep.error(log, "row_counts_reconciliation", f"discrepancias de filas: {bad}")
    else:
        log.info(f"  ✓ Sin discrepancias en {len(files)} examen(es) (total filas: {len(master)})")

    log.info("[validación] question_id_unique / numbering ...")
    rep.checks.append("question_id_unique")
    if master["question_id"].duplicated().any():
        rep.error(log, "question_id_unique", "question_id duplicados")
    else:
        log.info("  ✓ question_id único")

    log.info("[validación] correct_answer_values ...")
    rep.checks.append("correct_answer_values")
    valid = master.apply(lambda r: r["correct_answer"] in list("ABCDE")[: r["n_options_original"]], axis=1)
    if (~valid).any():
        rep.error(log, "correct_answer_values", f"{(~valid).sum()} respuesta(s) inválida(s)",
                  rows=master.loc[~valid, "question_id"].tolist())
    else:
        log.info("  ✓ Todas las respuestas son válidas para su número de alternativas (E nunca en exámenes de 4)")

    log.info("[validación] empty_critical_cells ...")
    rep.checks.append("empty_critical_cells")
    n_empty = 0
    for _, r in master.iterrows():
        cols = ["questions"] + [f"option_{l}" for l in "ABCDE"[: r["n_options_original"]]]
        n_empty += sum(1 for c in cols if not isinstance(r[c], str) or not r[c].strip())
    if n_empty:
        rep.error(log, "empty_critical_cells", f"{n_empty} celda(s) vacía(s) en columnas críticas")
    else:
        log.info("  ✓ Sin celdas vacías en columnas críticas")

    log.info("[validación] duplicate_questions_within_exam ...")
    rep.checks.append("duplicate_questions_within_exam")
    key = master["questions"].map(norm_text)
    dups = master[key.duplicated(keep=False).values].assign(_k=key)
    within = dups.groupby(["year", "exam", "_k"]).filter(lambda g: len(g) > 1)
    if len(within):
        for (y, e, _), g in within.groupby(["year", "exam", "_k"]):
            same_opts = g[OPTION_COLS].map(lambda v: norm_text(v) if isinstance(v, str) else "").drop_duplicates()
            rep.warn(log, "duplicate_questions_within_exam",
                     f"{y} {e}: enunciado repetido en P{g['question_number'].tolist()} "
                     f"({'mismas' if len(same_opts) == 1 else 'distintas'} alternativas)")
    else:
        log.info("  ✓ Sin duplicados dentro de cada examen")

    log.info("[validación] duplicate_exams_across_files ...")
    rep.checks.append("duplicate_exams_across_files")
    sig = master.groupby(["year", "exam"]).apply(
        lambda g: hash(tuple(g.sort_values("question_number")[["questions", *OPTION_COLS, "correct_answer"]]
                             .fillna("").map(norm_text).itertuples(index=False))), include_groups=False)
    for h, grp in sig.groupby(sig):
        if len(grp) > 1:
            rep.warn(log, "duplicate_exams_across_files",
                     f"exámenes con contenido idéntico (se conservan todos): {list(grp.index)}")

    log.info("[validación] duplicate_questions_across_exams ...")
    rep.checks.append("duplicate_questions_across_exams")
    k2 = master.assign(_k=key).groupby("_k")["question_id"].agg(list)
    cross = k2[k2.map(len) > 1]
    cross = cross[cross.map(lambda ids: len({i.rsplit('_', 1)[0] for i in ids}) > 1)]
    log.info(f"  ℹ {len(cross)} enunciado(s) aparecen en más de un examen (informativo; ver JSON de validación)")
    rep.warnings.append({"check": "duplicate_questions_across_exams", "message": f"{len(cross)} enunciados compartidos",
                         "groups": [ids for ids in cross.tolist()]})


def write_outputs(df: pd.DataFrame, out_dir: Path, name: str, log: logging.Logger) -> list[Path]:
    paths = []
    p_csv = out_dir / f"{name}.csv"
    df.to_csv(p_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")
    p_json = out_dir / f"{name}.json"
    recs = json.loads(df.to_json(orient="records", force_ascii=False))
    with open(p_json, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=1)
    p_pq = out_dir / f"{name}.parquet"
    df.to_parquet(p_pq, index=False)
    for p in (p_csv, p_json, p_pq):
        log.info(f"  ✓ {p.name:50s} {p.stat().st_size / 1e6:6.2f} MB")
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logger(stamp)
    rep = Report()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(d for d in BASE_DIR.glob("Year*") if d.is_dir() and re.fullmatch(r"Year\d{4}", d.name))
    log.info("=" * 100)
    log.info("Consolidación de exámenes CONAREME")
    log.info(f"Carpeta base    : {BASE_DIR}")
    log.info(f"Carpetas        : {[d.name for d in year_dirs]}")
    log.info(f"Salida          : {out_dir}")
    log.info("=" * 100)

    rows, files = [], []
    for yd in year_dirs:
        jsons = sorted(yd.glob("*.json"), key=lambda p: unicodedata.normalize("NFC", p.name))
        pdfs = list(yd.glob("*.pdf"))
        log.info(f"[{yd.name}] — {len(jsons)} JSON encontrado(s) ({len(pdfs)} PDF)")
        if len(jsons) != len(pdfs):
            rep.error(log, "files", f"{yd.name}: {len(jsons)} JSON vs {len(pdfs)} PDF")
        for jp in jsons:
            r, info = load_exam(jp, rep, log)
            rows.extend(r); files.append(info)
            log.info(f"  ✓ {info['exam']:40s} | filas: {info['rows']:4d} | opciones: {info['n_options']} | "
                     f"CSV=JSON: {info['csv_matches_json']}")

    original = pd.DataFrame(rows, columns=COLUMNS)
    original["year"] = original["year"].astype("int64")
    five = original.copy()
    mask4 = five["n_options_original"] == 4
    five.loc[mask4, "option_E"] = FILLER_E

    log.info("-" * 100)
    log.info(f"Total de filas    : {len(original):,}")
    log.info(f"Total de columnas : {len(COLUMNS)}")
    log.info(f"Columnas          : {COLUMNS}")
    log.info("Distribución por año:")
    for y, n in original.groupby("year").size().items():
        log.info(f"  • {y}  {n:5d} preguntas  ({original[original.year == y].exam.nunique()} exámenes)")
    log.info("Distribución por especialidad:")
    for s, n in original["specialty"].value_counts().items():
        log.info(f"  • {s:40s} {n:5d}")
    log.info(f"Preguntas con 4 alternativas originales: {int(mask4.sum()):,} | con 5: {int((~mask4).sum()):,}")
    log.info(f"Distribución de respuestas: {dict(sorted(Counter(original.correct_answer).items()))}")

    log.info("-" * 100)
    validate(original, files, rep, log)

    log.info("[validación] variants_consistency ...")
    rep.checks.append("variants_consistency")
    other = [c for c in COLUMNS if c != "option_E"]
    same_other = original[other].equals(five[other])
    e_ok5 = five.loc[~mask4, "option_E"].equals(original.loc[~mask4, "option_E"]) and (five.loc[mask4, "option_E"] == FILLER_E).all()
    e_ok_orig = original.loc[mask4, "option_E"].isna().all() and original.loc[~mask4, "option_E"].notna().all()
    if same_other and e_ok5 and e_ok_orig and five["option_E"].notna().all():
        log.info(f"  ✓ Variantes idénticas salvo option_E; variante 5 opciones sin option_E vacía; "
                 f"variante original con option_E vacía en exactamente {int(mask4.sum()):,} filas")
    else:
        rep.error(log, "variants_consistency", "las variantes no son consistentes")

    log.info("-" * 100)
    log.info("Escritura:")
    write_outputs(five, out_dir, "conareme_master_5opciones", log)
    write_outputs(original, out_dir, "conareme_master_opciones_originales", log)

    # Re-lectura de los CSV escritos (asegura que "Ninguna de las anteriores" y los vacíos se lean bien)
    r5 = pd.read_csv(out_dir / "conareme_master_5opciones.csv", dtype=str, encoding="utf-8-sig")
    ro = pd.read_csv(out_dir / "conareme_master_opciones_originales.csv", dtype=str, encoding="utf-8-sig")
    if r5["option_E"].isna().any() or ro["option_E"].isna().sum() != int(mask4.sum()) or len(r5) != len(original):
        rep.error(log, "reread", "la relectura de los CSV no coincide con lo esperado")
    else:
        log.info("  ✓ Relectura de CSV con pandas por defecto: OK")

    report = {
        "report": {"timestamp": datetime.now().isoformat(), "errors": rep.errors, "warnings": rep.warnings,
                   "checks_run": rep.checks,
                   "summary": {"total_rows": len(original), "total_exams": len(files),
                               "rows_4_options": int(mask4.sum()), "rows_5_options": int((~mask4).sum()),
                               "n_errors": len(rep.errors), "n_warnings": len(rep.warnings)}},
        "files_tracking": files,
    }
    vpath = out_dir / "conareme_master_validation.json"
    with open(vpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    log.info(f"  ✓ Reporte de validación: {vpath.name}")
    log.info(f"Validaciones: {len(rep.errors)} error(es), {len(rep.warnings)} advertencia(s)")
    log.info("✓ Proceso completado" + (" con errores" if rep.errors else ""))
    return 2 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
