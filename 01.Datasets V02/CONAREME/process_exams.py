#!/usr/bin/env python3
"""
process_exams.py — Extracción unificada de los exámenes CONAREME (2018–2026).

Reemplaza los ~100 notebooks (uno por examen). Para cada PDF en YearXXXX/:
  1. Extrae preguntas y alternativas recorriendo todo el documento (no página por
     página), con soporte para opciones en dos columnas, opciones de varias líneas
     y preguntas que cruzan de página.
  2. Detecta la alternativa correcta (resaltado amarillo) con DOS métodos
     independientes que deben coincidir:
        (a) rectángulos vectoriales amarillos (page.get_drawings())
        (b) proporción de píxeles amarillos en la imagen renderizada
  3. Verifica que no se pierde información (multiconjunto de caracteres del cuerpo
     del PDF == multiconjunto de caracteres de la salida).
  4. Compara contra el CSV verificado por humano (o su respaldo) antes de
     sobrescribirlo: respuesta correcta y similitud del texto de cada campo.
  5. Aplica correcciones humanas de answer_overrides.csv.
  6. Escribe <examen>.csv y <examen>.json junto al PDF, y genera log + reportes.

Uso:
    python3 process_exams.py                    # procesa todo y escribe
    python3 process_exams.py --dry-run          # solo reportes, no escribe CSV/JSON
    python3 process_exams.py --years 2024 2025 --exam "CIRUGÍA"
    python3 process_exams.py --render-review    # PNG de cada pregunta marcada
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import logging
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
BACKUP_DIR = BASE_DIR / "_backup_csv_verificados"
OVERRIDES_FILE = BASE_DIR / "answer_overrides.csv"

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
VEC_THRESHOLD = 0.50        # solapamiento mínimo (vertical × horizontal) con un rect. amarillo
PIX_THRESHOLD = 0.08        # fracción mínima de píxeles amarillos en la franja central de la opción
RENDER_ZOOM = 2.0
ROW_TOL = 3.0               # tolerancia vertical (pt) para considerar fragmentos en la misma fila
FRAGMENT_GAP = 40.0         # separación horizontal máxima para unir fragmentos de una línea
COLUMN_TOL = 6.0            # tolerancia de x0 entre marcadores de opción de una misma columna
TEXT_MINOR_RATIO = 0.97

QUESTION_RE = re.compile(r"^\s*(\d{1,3})\.(?!\d)\s*(.*)$", re.DOTALL)
OPTION_RE = re.compile(r"^\s*([A-E])\.\s*(.*)$", re.DOTALL)
UNDERSCORE_RE = re.compile(r"^[\s_]+$")


def exam_header_title(doc: fitz.Document) -> tuple[str | None, str | None]:
    """Título oficial del encabezado: ('SubEspecialidad', 'Cirugía General') o ('Especialidad', 'Prueba A')."""
    for page in list(doc)[:2]:
        text = page.get_text("text")
        m = re.search(r"Residentado\s+M.dico\s+\d{4}\s+((?:Sub)?Especialidad)\s*[-–]\s*([^\n]+)", text)
        if m:
            return m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
    return None, None


def expected_letters_for_year(year: int) -> list[str]:
    return list("ABCDE") if year <= 2022 else list("ABCD")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", nfc(str(s))).strip()


def is_yellow_fill(fill) -> bool:
    return bool(fill) and len(fill) >= 3 and fill[0] > 0.9 and fill[1] > 0.85 and fill[2] < 0.3


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------
@dataclass
class Line:
    page: int
    rect: fitz.Rect
    text: str


@dataclass
class Element:
    """Un enunciado o una alternativa: texto + líneas físicas de origen."""
    lines: list[Line] = field(default_factory=list)

    def add(self, line: Line, text: str | None = None):
        self.lines.append(Line(line.page, fitz.Rect(line.rect), line.text if text is None else text))

    @property
    def text(self) -> str:
        out = ""
        for ln in self.lines:
            t = nfc(ln.text).strip()
            if not t:
                continue
            if not out:
                out = t
            elif out.endswith("-") and t[:1].islower():
                out = out + t          # palabra cortada con guion: se une sin espacio (se conserva el guion)
            else:
                out = out + " " + t
        return re.sub(r"\s+", " ", out).strip()


@dataclass
class Question:
    number: int
    stem: Element = field(default_factory=Element)
    options: dict[str, Element] = field(default_factory=dict)
    option_col_x: list[float] = field(default_factory=list)
    pending: list[Line] = field(default_factory=list)
    correct: str | None = None
    vec_scores: dict[str, float] = field(default_factory=dict)
    pix_scores: dict[str, float] = field(default_factory=dict)
    has_image: bool = False
    flags: list[str] = field(default_factory=list)

    def all_lines(self) -> list[Line]:
        out = list(self.stem.lines)
        for e in self.options.values():
            out.extend(e.lines)
        return out


@dataclass
class ExamResult:
    year: int
    exam: str
    pdf: Path
    questions: list[Question] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_rows: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    preamble: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Lectura de líneas del cuerpo del PDF
# ---------------------------------------------------------------------------
def body_bounds(page: fitz.Page, raw_lines: list[Line]) -> tuple[float, float, bool]:
    """Límites verticales del cuerpo según las líneas de guiones bajos del encabezado/pie."""
    h = page.rect.height
    top, bottom, found = 0.0, h, False
    for ln in raw_lines:
        t = ln.text.strip()
        if len(t) >= 30 and UNDERSCORE_RE.match(t):
            if ln.rect.y1 < h * 0.2:
                top = max(top, ln.rect.y1); found = True
            elif ln.rect.y0 > h * 0.8:
                bottom = min(bottom, ln.rect.y0); found = True
    return top, bottom, found


def page_body_lines(page: fitz.Page, page_num: int) -> tuple[list[Line], fitz.Rect, bool]:
    raw: list[Line] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                raw.append(Line(page_num, fitz.Rect(line["bbox"]), text))
    top, bottom, found = body_bounds(page, raw)
    body = [ln for ln in raw if ln.rect.y0 >= top - 0.5 and ln.rect.y1 <= bottom + 0.5
            and not (len(ln.text.strip()) >= 30 and UNDERSCORE_RE.match(ln.text.strip()))]

    # Agrupar en filas (solapamiento vertical) y ordenar por x dentro de la fila.
    body.sort(key=lambda l: (l.rect.y0, l.rect.x0))
    rows: list[list[Line]] = []
    for ln in body:
        cy = (ln.rect.y0 + ln.rect.y1) / 2
        if rows:
            ref = rows[-1]
            ry0 = min(l.rect.y0 for l in ref); ry1 = max(l.rect.y1 for l in ref)
            if abs(ln.rect.y0 - ref[0].rect.y0) <= ROW_TOL or (ry0 <= cy <= ry1 and ln.rect.height < ref[0].rect.height):
                ref.append(ln)
                continue
        rows.append([ln])

    merged: list[Line] = []
    for row in rows:
        row.sort(key=lambda l: l.rect.x0)
        cur: Line | None = None
        for ln in row:
            starts_marker = bool(OPTION_RE.match(ln.text)) or bool(QUESTION_RE.match(ln.text))
            if cur is not None and 0 - 2 <= ln.rect.x0 - cur.rect.x1 < FRAGMENT_GAP and not starts_marker:
                gap = ln.rect.x0 - cur.rect.x1
                sep = "" if gap < 1.5 and not cur.text.endswith(" ") and not ln.text.startswith(" ") else " "
                cur = Line(page_num, cur.rect | ln.rect, cur.text.rstrip() + sep + ln.text.lstrip()
                           if sep else cur.text + ln.text)
            else:
                if cur is not None:
                    merged.append(cur)
                cur = Line(page_num, fitz.Rect(ln.rect), ln.text)
        if cur is not None:
            merged.append(cur)
    body_rect = fitz.Rect(0, top, page.rect.width, bottom)
    return merged, body_rect, found


# ---------------------------------------------------------------------------
# 2. Máquina de estados sobre todo el documento
# ---------------------------------------------------------------------------
def parse_questions(doc: fitz.Document, letters: list[str], res: ExamResult) -> list[Question]:
    questions: list[Question] = []
    cur: Question | None = None
    in_options = False

    for pno, page in enumerate(doc):
        lines, _, found = page_body_lines(page, pno)
        if not found:
            res.warnings.append(f"página {pno + 1}: no se encontraron líneas de encabezado/pie (____)")
        for ln in lines:
            text = ln.text
            # ¿Inicio de pregunta? Solo si es el número esperado y está en el margen izquierdo.
            mq = QUESTION_RE.match(text)
            expected_num = (cur.number + 1) if cur else 1
            if mq and int(mq.group(1)) == expected_num and (cur is None or in_options or not cur.stem.lines):
                cur = Question(number=expected_num)
                questions.append(cur)
                in_options = False
                cur.stem.add(ln, mq.group(2))
                continue
            if cur is None:
                res.preamble.append(text.strip())
                continue

            # ¿Inicio de alternativa? Letra esperada aún no vista, en una columna coherente.
            mo = OPTION_RE.match(text)
            if mo and mo.group(1) in letters and mo.group(1) not in cur.options:
                letter = mo.group(1)
                x0 = ln.rect.x0
                col_ok = (not cur.option_col_x or any(abs(x0 - cx) <= COLUMN_TOL for cx in cur.option_col_x)
                          or all(abs(x0 - cx) > 100 for cx in cur.option_col_x))
                if col_ok:
                    if not any(abs(x0 - cx) <= COLUMN_TOL for cx in cur.option_col_x):
                        cur.option_col_x.append(x0)
                    el = Element()
                    el.add(ln, mo.group(2))
                    cur.options[letter] = el
                    in_options = True
                    continue

            # Continuación.
            if not in_options:
                cur.stem.add(ln)
            else:
                cur.pending.append(ln)   # se asigna al cerrar la pregunta, cuando ya se conocen todas las alternativas
                if OPTION_RE.match(text) or QUESTION_RE.match(text):
                    cur.flags.append(f"línea de continuación con forma de marcador: {text.strip()[:60]!r}")
    for q in questions:
        _resolve_pending(q)
    return questions


def _resolve_pending(q: Question):
    """Asigna las líneas de continuación de la zona de alternativas.

    Se recorren en orden de lectura por columnas (página, columna, y), como fluye el texto en Word:
    primero la columna izquierda completa y luego la derecha."""
    if not q.pending:
        return
    cols = sorted(q.option_col_x)

    def col_idx(x: float) -> int:
        idx = 0
        for i, cx in enumerate(cols):
            if x >= cx - COLUMN_TOL:
                idx = i
        return idx

    def reading_key(el: Element):
        f = el.lines[0]
        return (f.page, col_idx(f.rect.x0), f.rect.y0)

    for ln in sorted(q.pending, key=lambda l: (l.page, col_idx(l.rect.x0), l.rect.y0)):
        ci = col_idx(ln.rect.x0)
        target, how = None, None
        # (1) misma columna y página: la alternativa con la línea más cercana por encima
        best_y = None
        for el in q.options.values():
            if el.lines[0].page != ln.page or col_idx(el.lines[0].rect.x0) != ci:
                continue
            ys = [l.rect.y0 for l in el.lines if l.page == ln.page and l.rect.y0 < ln.rect.y0 - 1]
            if ys and (best_y is None or max(ys) > best_y):
                best_y, target = max(ys), el
        if target is not None:
            how = "columna"
        # (2) enunciado justo encima, alineado a su margen (enunciado intercalado con las alternativas)
        if target is None and q.stem.lines:
            s_last = q.stem.lines[-1]
            s_left = min(l.rect.x0 for l in q.stem.lines if l.page == s_last.page)
            if (s_last.page == ln.page and -2 <= ln.rect.y0 - s_last.rect.y1 < 4
                    and abs(ln.rect.x0 - s_left) <= 20):
                target, how = q.stem, "enunciado"
        # (3) arriba de una columna derecha: continúa la última alternativa de la columna anterior
        if target is None and ci > 0:
            prev = [el for el in q.options.values()
                    if el.lines[0].page == ln.page and col_idx(el.lines[0].rect.x0) == ci - 1]
            if prev:
                target, how = max(prev, key=lambda el: el.lines[0].rect.y0), "flujo_columna"
        # (4) inicio de página: continúa la última alternativa (orden de lectura) de páginas anteriores
        if target is None:
            prev = [el for el in q.options.values() if el.lines[0].page < ln.page]
            if prev:
                target, how = max(prev, key=reading_key), "cambio_pagina"
        if target is None:
            target, how = list(q.options.values())[-1], "respaldo"
            q.flags.append(f"continuación asignada por respaldo: {ln.text.strip()[:60]!r}")
        target.add(ln)
        if how == "enunciado" and "enunciado_intercalado_con_alternativas" not in q.flags:
            q.flags.append("enunciado_intercalado_con_alternativas")
        if how == "flujo_columna" and "alternativa_continua_en_otra_columna" not in q.flags:
            q.flags.append("alternativa_continua_en_otra_columna")
    q.pending = []


# ---------------------------------------------------------------------------
# 3. Detección de la respuesta correcta
# ---------------------------------------------------------------------------
def yellow_rects_by_page(doc: fitz.Document) -> dict[int, list[fitz.Rect]]:
    out: dict[int, list[fitz.Rect]] = {}
    for pno, page in enumerate(doc):
        rects = []
        for d in page.get_drawings():
            if is_yellow_fill(d.get("fill")):
                r = fitz.Rect(d["rect"])
                if r.width > 1 and r.height > 1:
                    rects.append(r)
        out[pno] = rects
    return out


def vec_score(line: Line, yrects: list[fitz.Rect]) -> float:
    """Máximo, sobre los rectángulos amarillos, de (solapamiento vertical relativo × horizontal relativo).

    Se normaliza por la dimensión menor (línea o rectángulo) para que funcione tanto con resaltados
    que cubren solo el texto corto de la opción (p. ej. "A. 2") como con resaltados que ocupan
    todo el ancho de la columna."""
    r = line.rect
    if r.is_empty or r.width <= 0 or r.height <= 0:
        return 0.0
    best = 0.0
    for y in yrects:
        if not r.intersects(y):
            continue
        i = r & y
        fv = i.height / max(1e-6, min(r.height, y.height))
        fh = i.width / max(1e-6, min(r.width, y.width))
        best = max(best, min(1.0, fv) * min(1.0, fh))
    return best


def pix_score(page: fitz.Page, rect: fitz.Rect) -> float:
    """Fracción de píxeles amarillos en la franja central (25–75 % de la altura) de la línea renderizada.

    La franja central evita contar el borde de un resaltado de la línea vecina."""
    clip = fitz.Rect(rect)
    if clip.is_empty:
        return 0.0
    pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    h = img.shape[0]
    band = img[int(h * 0.25): max(int(h * 0.75), int(h * 0.25) + 1)]
    yellow = (band[:, :, 0] > 200) & (band[:, :, 1] > 200) & (band[:, :, 2] < 150)
    return float(yellow.mean())


def detect_answers(doc: fitz.Document, questions: list[Question], res: ExamResult):
    yrects = yellow_rects_by_page(doc)
    n_agree = 0
    for q in questions:
        for letter, el in q.options.items():
            first = el.lines[0]
            q.vec_scores[letter] = round(vec_score(first, yrects[first.page]), 4)
            q.pix_scores[letter] = round(pix_score(doc[first.page], first.rect), 4)
        vec_hits = [l for l, s in q.vec_scores.items() if s >= VEC_THRESHOLD]
        pix_hits = [l for l, s in q.pix_scores.items() if s >= PIX_THRESHOLD]
        if len(vec_hits) == 1 and vec_hits == pix_hits:
            q.correct = vec_hits[0]
            n_agree += 1
        else:
            q.correct = vec_hits[0] if len(vec_hits) == 1 else (pix_hits[0] if len(pix_hits) == 1 else None)
            msg = (f"P{q.number}: detección de respuesta ambigua/discordante "
                   f"vec={q.vec_scores} pix={q.pix_scores} -> {q.correct}")
            q.flags.append("respuesta_ambigua")
            res.errors.append(msg)
            res.review_rows.append(_review(res, q, "correct_answer", "", q.correct or "", None,
                                           "detección de resaltado ambigua o métodos en desacuerdo"))

    # Cada rectángulo amarillo se asigna a la línea con la que más se solapa.
    owners = []  # (page, Line, question, kind, letter)
    for q in questions:
        for ln in q.stem.lines:
            owners.append((ln, q, "stem", None))
        for l, el in q.options.items():
            for ln in el.lines:
                owners.append((ln, q, "option", l))
    by_page: dict[int, list] = {}
    for o in owners:
        by_page.setdefault(o[0].page, []).append(o)
    orphans = 0
    blank_hl = 0
    for p, rects in yrects.items():
        for r in rects:
            best, best_a = None, 0.0
            for o in by_page.get(p, []):
                if o[0].rect.intersects(r):
                    a = (o[0].rect & r).get_area()
                    if a > best_a:
                        best, best_a = o, a
            # descartar contactos de borde (< 15 % de la altura del rectángulo)
            if best is not None and (best[0].rect & r).height < 0.15 * r.height:
                best = None
            if best is None:
                words = [w for w in doc[p].get_text("words") if fitz.Rect(w[:4]).intersects(r)
                         and (fitz.Rect(w[:4]) & r).height > 0.5 * fitz.Rect(w[:4]).height]
                if not words:
                    blank_hl += 1  # resaltado sobre una línea en blanco: sin efecto sobre el contenido
                    continue
                orphans += 1
                res.warnings.append(f"resaltado amarillo huérfano en pág. {p + 1} {tuple(round(v) for v in r)} "
                                    f"sobre: {' '.join(w[4] for w in words)[:60]!r}")
            elif best[2] == "stem":
                res.warnings.append(f"resaltado amarillo sobre el enunciado de P{best[1].number} (pág. {p + 1})")
                res.review_rows.append(_review(res, best[1], "correct_answer", "", best[1].correct or "", None,
                                               "resaltado amarillo sobre el enunciado"))
            elif best[3] != best[1].correct:
                res.warnings.append(f"P{best[1].number}: resaltado sobre la alternativa {best[3]} "
                                    f"distinta de la respuesta detectada {best[1].correct} (pág. {p + 1})")
                res.review_rows.append(_review(res, best[1], "correct_answer", "", best[1].correct or "", None,
                                               f"resaltado adicional sobre la alternativa {best[3]}"))
    res.verification["answer_methods_agree"] = n_agree
    res.verification["orphan_highlights"] = orphans
    res.verification["blank_line_highlights"] = blank_hl

    # estadísticas de margen
    win_v = [q.vec_scores[q.correct] for q in questions if q.correct]
    lose_v = [s for q in questions for l, s in q.vec_scores.items() if l != q.correct]
    win_p = [q.pix_scores[q.correct] for q in questions if q.correct]
    lose_p = [s for q in questions for l, s in q.pix_scores.items() if l != q.correct]
    res.verification["score_margins"] = {
        "vec_min_winner": min(win_v) if win_v else None, "vec_max_loser": max(lose_v) if lose_v else None,
        "pix_min_winner": min(win_p) if win_p else None, "pix_max_loser": max(lose_p) if lose_p else None,
    }


# ---------------------------------------------------------------------------
# 4. Imágenes
# ---------------------------------------------------------------------------
def mark_images(doc: fitz.Document, questions: list[Question], res: ExamResult):
    for pno, page in enumerate(doc):
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"])
            if r.width < 5 or r.height < 5:
                continue
            best, best_d = None, None
            for q in questions:
                for ln in q.all_lines():
                    if ln.page != pno:
                        continue
                    d = 0 if ln.rect.intersects(r) else max(0.0, r.y0 - ln.rect.y1)
                    if ln.rect.y0 <= r.y1 and (best_d is None or d < best_d):
                        best, best_d = q, d
            if best:
                best.has_image = True
                best.flags.append("contiene_imagen")
                res.warnings.append(f"P{best.number}: contiene una imagen (pág. {pno + 1}); verificar que el texto sea autosuficiente")
                res.review_rows.append(_review(res, best, "imagen", "", "", None, "pregunta con imagen"))
            else:
                res.warnings.append(f"imagen en pág. {pno + 1} no asociada a ninguna pregunta")


# ---------------------------------------------------------------------------
# 5. Validaciones estructurales y de cobertura
# ---------------------------------------------------------------------------
def structural_checks(questions: list[Question], letters: list[str], res: ExamResult):
    nums = [q.number for q in questions]
    if nums != list(range(1, len(nums) + 1)):
        res.errors.append(f"numeración no consecutiva: {nums}")
    if len(questions) not in (90, 100):
        res.errors.append(f"número de preguntas inesperado: {len(questions)}")
    if res.preamble:
        res.warnings.append(f"{len(res.preamble)} línea(s) antes de la pregunta 1: {res.preamble[:3]}")
    seen = {}
    for q in questions:
        if sorted(q.options) != letters:
            res.errors.append(f"P{q.number}: alternativas encontradas {sorted(q.options)} (esperadas {letters})")
        if not q.stem.text:
            res.errors.append(f"P{q.number}: enunciado vacío")
        for l, e in q.options.items():
            if not e.text:
                res.errors.append(f"P{q.number}: alternativa {l} vacía")
        if re.match(r"^\d{1,3}\.\s", q.stem.text):
            res.warnings.append(f"P{q.number}: el enunciado empieza con un número residual: {q.stem.text[:30]!r}")
        for f in q.flags:
            if f.startswith("línea de continuación") or f.startswith("continuación asignada por respaldo"):
                res.warnings.append(f"P{q.number}: {f}")
        key = norm_ws(q.stem.text).lower()
        if key in seen:
            res.warnings.append(f"P{q.number}: enunciado idéntico a P{seen[key]} (duplicado dentro del examen)")
            res.review_rows.append(_review(res, q, "questions", "", q.stem.text, None,
                                           f"enunciado duplicado de P{seen[key]} (posible error del PDF original)"))
            q.flags.append(f"duplicado_de_P{seen[key]}")
        else:
            seen[key] = q.number


def coverage_check(doc: fitz.Document, questions: list[Question], res: ExamResult) -> bool:
    """Multiconjunto de caracteres no-blancos: palabras del cuerpo del PDF vs salida."""
    pdf_chars: Counter = Counter()
    for pno, page in enumerate(doc):
        lines, body_rect, _ = page_body_lines(page, pno)
        for w in page.get_text("words"):
            r = fitz.Rect(w[:4])
            cy = (r.y0 + r.y1) / 2
            if body_rect.y0 <= cy <= body_rect.y1:
                t = nfc(w[4])
                if len(t) >= 30 and UNDERSCORE_RE.match(t):
                    continue
                pdf_chars.update(c for c in t if not c.isspace())
    out_chars: Counter = Counter()
    for q in questions:
        out_chars.update(c for c in f"{q.number}." if not c.isspace())
        out_chars.update(c for c in q.stem.text if not c.isspace())
        for l, e in q.options.items():
            out_chars.update(c for c in f"{l}." if not c.isspace())
            out_chars.update(c for c in e.text if not c.isspace())
    for t in res.preamble:
        out_chars.update(c for c in nfc(t) if not c.isspace())
    missing = pdf_chars - out_chars
    extra = out_chars - pdf_chars
    ok = not missing and not extra
    res.verification["coverage_ok"] = ok
    res.verification["coverage_pdf_chars"] = sum(pdf_chars.values())
    res.verification["coverage_out_chars"] = sum(out_chars.values())
    if not ok:
        res.errors.append(f"cobertura: faltan {dict(missing)} / sobran {dict(extra)}")
    return ok


# ---------------------------------------------------------------------------
# 6. Comparación con el CSV verificado por humano
# ---------------------------------------------------------------------------
def classify(old: str, new: str) -> tuple[str, float]:
    if old == new:
        return "identico", 1.0
    o, n = norm_ws(old), norm_ws(new)
    if o == n:
        return "solo_espacios", 1.0
    if re.sub(r"\s+", "", o) == re.sub(r"\s+", "", n):
        return "solo_espacios", 1.0
    ratio = difflib.SequenceMatcher(None, o, n, autojunk=False).ratio()
    if n.startswith(o) and len(n) > len(o):
        return "csv_antiguo_truncado", ratio
    if ratio >= TEXT_MINOR_RATIO:
        return "menor", ratio
    return "mayor", ratio


def compare_with_human(questions: list[Question], letters: list[str], ref_csv: Path | None, res: ExamResult):
    v = res.verification
    if ref_csv is None or not ref_csv.exists():
        res.warnings.append("no existe CSV humano de referencia")
        v["human_reference"] = None
        return
    df = pd.read_csv(ref_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    v["human_reference"] = str(ref_csv.relative_to(BASE_DIR))
    v["human_rows"] = len(df)
    if len(df) != len(questions):
        res.errors.append(f"filas: CSV humano={len(df)} vs extraídas={len(questions)}")
    n = min(len(df), len(questions))
    matches, mism = 0, []
    cats: Counter = Counter()
    for i in range(n):
        q, row = questions[i], df.iloc[i]
        old_ans = row.get("correct_answer", "").strip()
        if old_ans == (q.correct or ""):
            matches += 1
        else:
            mism.append(q.number)
            q.flags.append(f"discrepancia_respuesta_humano={old_ans}")
            res.review_rows.append(_review(res, q, "correct_answer", old_ans, q.correct or "", None,
                                           "respuesta del PDF distinta a la del CSV verificado por humano"))
        fields = [("questions", q.stem.text)] + [(f"option_{l}", q.options[l].text if l in q.options else "")
                                                  for l in letters]
        for col, new in fields:
            old = row.get(col, "")
            cat, ratio = classify(old, new)
            cats[cat] += 1
            if cat in ("mayor", "csv_antiguo_truncado", "menor"):
                res.review_rows.append(_review(res, q, col, old, new, round(ratio, 4), f"texto: {cat}"))
    v["answers_match_human"] = matches
    v["answers_compared"] = n
    v["answer_mismatches"] = mism
    v["text_diff_categories"] = dict(cats)
    for qn in mism:
        res.warnings.append(f"P{qn}: DISCREPANCIA de respuesta con el CSV humano (ver revisión)")


def _review(res: ExamResult, q: Question, fld: str, old, new, ratio, reason: str) -> dict:
    pages = sorted({ln.page + 1 for ln in q.all_lines()})
    return {"year": res.year, "exam": res.exam, "question_number": q.number, "field": fld,
            "valor_csv_humano": old, "valor_extraido": new, "similitud": ratio, "motivo": reason,
            "paginas_pdf": ",".join(map(str, pages))}


# ---------------------------------------------------------------------------
# 7. Correcciones humanas
# ---------------------------------------------------------------------------
def load_overrides() -> pd.DataFrame:
    if not OVERRIDES_FILE.exists():
        return pd.DataFrame(columns=["year", "exam", "question_number", "field", "value", "reason"])
    return pd.read_csv(OVERRIDES_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def apply_overrides(questions: list[Question], ov: pd.DataFrame, res: ExamResult, log: logging.Logger):
    if ov.empty:
        return 0
    sel = ov[(ov["year"].str.strip() == str(res.year)) & (ov["exam"].map(lambda s: nfc(s.strip())) == nfc(res.exam))]
    by_num = {q.number: q for q in questions}
    n = 0
    for _, r in sel.iterrows():
        q = by_num.get(int(r["question_number"]))
        fld, val = r["field"].strip(), r["value"]
        if q is None:
            res.errors.append(f"override para P{r['question_number']} inexistente"); continue
        if fld == "correct_answer":
            old = q.correct; q.correct = val.strip()
        elif fld == "questions":
            old = q.stem.text; q.stem = Element([Line(-1, fitz.Rect(), val)])
        elif fld.startswith("option_") and fld[-1] in q.options:
            old = q.options[fld[-1]].text; q.options[fld[-1]] = Element([Line(-1, fitz.Rect(), val)])
        else:
            res.errors.append(f"override con campo inválido: {fld}"); continue
        q.flags.append(f"override:{fld}")
        log.info(f"    ✎ override P{q.number} {fld}: {old!r} -> {val!r} ({r.get('reason', '')})")
        n += 1
    return n


# ---------------------------------------------------------------------------
# 8. Escritura
# ---------------------------------------------------------------------------
def to_dataframe(questions: list[Question], letters: list[str]) -> pd.DataFrame:
    rows = []
    for q in questions:
        row = {"question_number": q.number, "questions": q.stem.text}
        for l in letters:
            row[f"option_{l}"] = q.options[l].text if l in q.options else ""
        row["correct_answer"] = q.correct or ""
        rows.append(row)
    return pd.DataFrame(rows, columns=["question_number", "questions"] + [f"option_{l}" for l in letters]
                        + ["correct_answer"])


def to_json(res: ExamResult, letters: list[str], sha: str) -> dict:
    return {
        "metadata": {
            "year": res.year, "exam": res.exam, "source_pdf": res.pdf.name,
            "exam_type": res.verification.get("exam_type"), "exam_title": res.verification.get("exam_title"),
            "sha256_pdf": sha, "n_questions": len(res.questions), "n_options": len(letters),
            "options": letters, "extracted_at": datetime.now().isoformat(timespec="seconds"),
            "script_version": SCRIPT_VERSION, "verification": res.verification,
        },
        "questions": [
            {"number": q.number, "question": q.stem.text,
             "options": {l: (q.options[l].text if l in q.options else "") for l in letters},
             "correct_answer": q.correct, "has_image": q.has_image, "flags": q.flags}
            for q in res.questions
        ],
    }


def render_review(doc: fitz.Document, res: ExamResult, out_dir: Path):
    flagged = {r["question_number"] for r in res.review_rows}
    by_num = {q.number: q for q in res.questions}
    for qn in sorted(flagged):
        q = by_num.get(qn)
        if not q:
            continue
        per_page: dict[int, fitz.Rect] = {}
        for ln in q.all_lines():
            if ln.page < 0:
                continue
            per_page[ln.page] = (per_page[ln.page] | ln.rect) if ln.page in per_page else fitz.Rect(ln.rect)
        for p, r in per_page.items():
            page = doc[p]
            clip = fitz.Rect(30, r.y0 - 4, page.rect.width - 30, r.y1 + 4) & page.rect
            pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2))
            safe = re.sub(r"[^\w]+", "_", res.exam)
            pix.save(out_dir / f"{res.year}_{safe}_P{qn:03d}_pag{p + 1}.png")


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def setup_logger(stamp: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    log = logging.getLogger("conareme")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / f"procesamiento_{stamp}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def process_pdf(pdf: Path, year: int, args, overrides: pd.DataFrame, log: logging.Logger,
                review_dir: Path | None) -> ExamResult:
    exam = nfc(pdf.stem)
    res = ExamResult(year=year, exam=exam, pdf=pdf)
    letters = expected_letters_for_year(year)
    doc = fitz.open(pdf)
    res.verification["pages"] = len(doc)
    etype, etitle = exam_header_title(doc)
    res.verification["exam_type"], res.verification["exam_title"] = etype, etitle
    if not etitle:
        res.warnings.append("no se pudo leer el título del examen en el encabezado")

    questions = parse_questions(doc, letters, res)
    res.questions = questions
    detect_answers(doc, questions, res)
    mark_images(doc, questions, res)
    structural_checks(questions, letters, res)
    coverage_check(doc, questions, res)

    # Detectar opciones auto (¿hay líneas "E." en exámenes de 4 opciones?)
    if len(letters) == 4:
        e_lines = sum(1 for q in questions for e in [q.stem, *q.options.values()]
                      for ln in e.lines[1:] if re.match(r"^\s*E\.\s", ln.text))
        if e_lines:
            res.errors.append(f"se hallaron {e_lines} línea(s) 'E.' en un examen de 4 opciones")

    # Referencia humana: el respaldo (si existe) tiene prioridad sobre el CSV actual.
    backup = BACKUP_DIR / pdf.parent.name / f"{pdf.stem}.csv"
    current = pdf.with_suffix(".csv")
    ref = backup if backup.exists() else (current if current.exists() else None)
    compare_with_human(questions, letters, ref, res)

    n_ov = apply_overrides(questions, overrides, res, log)
    res.verification["overrides_applied"] = n_ov

    if review_dir is not None:
        render_review(doc, res, review_dir)

    if not args.dry_run:
        if current.exists() and not backup.exists() and not args.no_backup:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, backup)
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        df = to_dataframe(questions, letters)
        df.to_csv(current, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")
        with open(pdf.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(to_json(res, letters, sha), fh, ensure_ascii=False, indent=2)
    doc.close()
    return res


def log_exam(res: ExamResult, log: logging.Logger):
    v = res.verification
    n = len(res.questions)
    log.info(f"  [{res.year}] {res.exam}")
    log.info(f"      preguntas: {n} | opciones: {len(expected_letters_for_year(res.year))} | "
             f"páginas: {v.get('pages')} | métodos de respuesta concordantes: {v.get('answer_methods_agree')}/{n}")
    if v.get("human_reference"):
        log.info(f"      respuestas = CSV humano: {v.get('answers_match_human')}/{v.get('answers_compared')} | "
                 f"texto vs humano: {v.get('text_diff_categories')}")
    log.info(f"      cobertura de texto: {'OK' if v.get('coverage_ok') else 'FALLA'} "
             f"({v.get('coverage_out_chars')}/{v.get('coverage_pdf_chars')} caracteres) | "
             f"márgenes: {v.get('score_margins')}")
    for e in res.errors:
        log.error(f"      ✗ {e}")
    for w in res.warnings:
        log.warning(f"      ⚠ {w}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="*", type=int, help="años a procesar (por defecto, todos)")
    ap.add_argument("--exam", help="procesar solo los PDFs cuyo nombre contenga este texto")
    ap.add_argument("--dry-run", action="store_true", help="no escribe CSV/JSON; solo log y reportes")
    ap.add_argument("--no-backup", action="store_true", help="no respaldar el CSV humano antes de sobrescribir")
    ap.add_argument("--render-review", action="store_true", help="PNG de cada pregunta marcada para revisión")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logger(stamp)
    review_dir = None
    if args.render_review:
        review_dir = LOG_DIR / f"review_png_{stamp}"
        review_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = sorted(d for d in BASE_DIR.glob("Year*") if d.is_dir() and re.fullmatch(r"Year\d{4}", d.name))
    if args.years:
        year_dirs = [d for d in year_dirs if int(d.name[4:]) in args.years]

    log.info("=" * 100)
    log.info(f"Procesamiento de exámenes CONAREME — process_exams.py v{SCRIPT_VERSION}")
    log.info(f"Carpeta base : {BASE_DIR}")
    log.info(f"Años         : {[d.name for d in year_dirs]}")
    log.info(f"Modo         : {'DRY-RUN (no se escribe)' if args.dry_run else 'ESCRITURA (CSV + JSON)'}")
    log.info("=" * 100)

    overrides = load_overrides()
    if not overrides.empty:
        log.info(f"Correcciones humanas cargadas: {len(overrides)} fila(s) de {OVERRIDES_FILE.name}")

    results: list[ExamResult] = []
    for yd in year_dirs:
        year = int(yd.name[4:])
        pdfs = sorted(yd.glob("*.pdf"), key=lambda p: nfc(p.name))
        if args.exam:
            pdfs = [p for p in pdfs if nfc(args.exam).lower() in nfc(p.stem).lower()]
        log.info(f"[{yd.name}] {len(pdfs)} PDF(s)")
        for pdf in pdfs:
            try:
                res = process_pdf(pdf, year, args, overrides, log, review_dir)
            except Exception as exc:  # noqa: BLE001
                log.exception(f"  ✗ {pdf.name}: fallo fatal: {exc}")
                res = ExamResult(year=year, exam=nfc(pdf.stem), pdf=pdf, errors=[f"fallo fatal: {exc}"])
            results.append(res)
            log_exam(res, log)

    # ---------------- Resumen global ----------------
    total_q = sum(len(r.questions) for r in results)
    agree = sum(r.verification.get("answer_methods_agree", 0) for r in results)
    match = sum(r.verification.get("answers_match_human", 0) for r in results)
    comp = sum(r.verification.get("answers_compared", 0) for r in results)
    cov_ok = sum(1 for r in results if r.verification.get("coverage_ok"))
    cats: Counter = Counter()
    for r in results:
        cats.update(r.verification.get("text_diff_categories", {}))
    n_err = sum(len(r.errors) for r in results)
    n_warn = sum(len(r.warnings) for r in results)
    log.info("=" * 100)
    log.info("RESUMEN")
    log.info(f"  Exámenes procesados          : {len(results)}")
    log.info(f"  Preguntas extraídas          : {total_q}")
    log.info(f"  Métodos de respuesta (a)=(b) : {agree}/{total_q}")
    log.info(f"  Respuestas = CSV humano      : {match}/{comp}")
    log.info(f"  Cobertura de texto OK        : {cov_ok}/{len(results)}")
    log.info(f"  Diferencias de texto vs humano: {dict(cats)}")
    log.info(f"  Errores: {n_err} | Advertencias: {n_warn}")
    for r in results:
        for qn in r.verification.get("answer_mismatches", []):
            log.warning(f"  ⚠ DISCREPANCIA respuesta: {r.year} {r.exam} P{qn}")

    review = [row for r in results for row in r.review_rows]
    review_path = LOG_DIR / f"revision_humana_{stamp}.csv"
    pd.DataFrame(review, columns=["year", "exam", "question_number", "field", "valor_csv_humano",
                                  "valor_extraido", "similitud", "motivo", "paginas_pdf"]
                 ).to_csv(review_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    report = {
        "timestamp": stamp, "script_version": SCRIPT_VERSION, "dry_run": args.dry_run,
        "summary": {"exams": len(results), "questions": total_q, "answer_methods_agree": agree,
                    "answers_match_human": match, "answers_compared": comp, "coverage_ok": cov_ok,
                    "text_diff_categories": dict(cats), "n_errors": n_err, "n_warnings": n_warn,
                    "n_review_rows": len(review)},
        "exams": [{"year": r.year, "exam": r.exam, "pdf": str(r.pdf.relative_to(BASE_DIR)),
                   "n_questions": len(r.questions), "verification": r.verification,
                   "errors": r.errors, "warnings": r.warnings} for r in results],
    }
    report_path = LOG_DIR / f"reporte_verificacion_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info(f"  Reporte JSON      : {report_path.relative_to(BASE_DIR)}")
    log.info(f"  Revisión humana   : {review_path.relative_to(BASE_DIR)} ({len(review)} fila(s))")
    if review_dir:
        log.info(f"  PNG de revisión   : {review_dir.relative_to(BASE_DIR)}")
    log.info("✓ Proceso completado" + (" con errores — revisar log" if n_err else ""))
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
