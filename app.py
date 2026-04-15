from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse

# -------------------- Configuration --------------------

API_KEY = os.getenv("API_KEY", "dev-key")
MAX_FILE_SIZE_MB = 15  # adjust if needed

MONTH_MAP = {
    "JAN": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARY": 2,
    "MAR": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5,
    "JUN": 6, "JUNE": 6,
    "JUL": 7, "JULY": 7,
    "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DECEMBER": 12,
}

SHIFT_MAP = {
    "D": ("Day Shift", "on_unit"),
    "N": ("Night Shift", "on_unit"),
    "OT": ("Overtime", "on_unit"),
    "ED": ("Education Day", "off_unit"),
    "UC": ("Unit Council", "off_unit"),
    "ILL": ("Sick Day", "off"),
    "VAC": ("Vacation", "off"),
    "SH": ("Stat Holiday", "off"),
    "FAMILY": ("Family Emergency", "off"),
}

DEFAULT_IGNORE_TOKENS = {
    "SS", "IC", "PU", "CAP", "X", "?", "M", "DAY", "NIGHT", "PCC",
    "FULL", "TIME", "NURSES", "PART", "CASUALNURSES", "CASUAL", "NURSES",
    "VAC-", "SH+",
}

# -------------------- Data structures --------------------

@dataclass
class Word:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

# -------------------- Utilities --------------------

def _clean_text(v: Optional[str]) -> str:
    return "" if v is None else str(v).strip()


def _is_day_token(s: str) -> bool:
    s = _clean_text(s)
    return s.isdigit() and 1 <= int(s) <= 31


def _normalize_name(raw_name: str) -> str:
    """Convert 'LAST, FIRST (notes) - MAT LEAVE' -> 'FIRST LAST'."""
    text = _clean_text(raw_name).upper()
    text = re.sub(r"\s+-\s+.*$", "", text)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if "," in text:
        last, first = [p.strip() for p in text.split(",", 1)]
        return re.sub(r"\s+", " ", f"{first} {last}").strip()
    return text


def _extract_year_from_pdf_text(full_text: str) -> Optional[int]:
    years = re.findall(r"\b(20\d{2})\b", full_text)
    years = [int(y) for y in years]
    if not years:
        return None
    counts: Dict[int, int] = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _cluster_by_y(words: List[Word], tol: float = 1.5) -> List[List[Word]]:
    """Cluster words into horizontal bands by y0."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.page, w.y0, w.x0))
    clusters: List[List[Word]] = []
    current: List[Word] = [words_sorted[0]]
    for w in words_sorted[1:]:
        last = current[-1]
        if w.page == last.page and abs(w.y0 - last.y0) <= tol:
            current.append(w)
        else:
            clusters.append(current)
            current = [w]
    clusters.append(current)
    return clusters


def _words_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[List[Word], str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_words: List[Word] = []
    text_chunks: List[str] = []
    for pi in range(doc.page_count):
        page = doc[pi]
        text_chunks.append(page.get_text("text"))
        for (x0, y0, x1, y1, txt, *_rest) in page.get_text("words"):
            t = _clean_text(txt)
            if t:
                all_words.append(Word(page=pi, x0=x0, y0=y0, x1=x1, y1=y1, text=t))
    return all_words, "\n".join(text_chunks)

# -------------------- Core parsing --------------------

def _detect_day_header(words: List[Word]) -> Tuple[List[Word], float]:
    """Find the day-number header row."""
    day_words = [w for w in words if _is_day_token(w.text)]
    if not day_words:
        raise ValueError("No day-number header row found in PDF.")
    clusters = _cluster_by_y(day_words, tol=1.2)
    best = max(clusters, key=lambda c: len(c))
    y = sum(w.y0 for w in best) / len(best)
    return sorted(best, key=lambda w: w.x0), y


def _detect_month_words(words: List[Word], day_header_y: float) -> List[Word]:
    month_words = []
    for w in words:
        if w.y0 < day_header_y - 1 and _clean_text(w.text).upper() in MONTH_MAP:
            month_words.append(w)
    month_words = [w for w in month_words if (day_header_y - w.y0) < 25]
    return sorted(month_words, key=lambda w: w.x0)


def _build_dates(
    day_tokens: List[Word],
    month_words: List[Word],
    full_text: str,
    year_hint: Optional[int],
    warnings: List[dict],
) -> List[Optional[date]]:
    day_nums = [int(w.text) for w in day_tokens]

    if month_words:
        start_month = MONTH_MAP[month_words[0].text.upper()]
    else:
        warnings.append({
            "type": "MONTH_HEADER_NOT_FOUND",
            "message": "Could not find month header row; defaulting to April."
        })
        start_month = 4

    detected_year = _extract_year_from_pdf_text(full_text)
    year = year_hint or detected_year or datetime.now().year
    if year_hint is None and detected_year is None:
        warnings.append({
            "type": "YEAR_NOT_FOUND",
            "message": f"No year found in PDF text; defaulting to {year}.",
            "year": year
        })

    dates: List[Optional[date]] = []
    month = start_month
    prev_day = None

    for d in day_nums:
        if prev_day is not None and d < prev_day:
            month += 1
            if month > 12:
                month = 1
                year += 1
        try:
            dates.append(date(year, month, d))
        except ValueError:
            warnings.append({"type": "INVALID_DATE", "day": d, "month": month, "year": year})
            dates.append(None)
        prev_day = d

    for i in range(1, len(dates)):
        if dates[i] is None or dates[i - 1] is None:
            continue
        delta = (dates[i] - dates[i - 1]).days
        if delta != 1:
            warnings.append({
                "type": "DATE_SEQUENCE_GAP",
                "index": i,
                "prev": dates[i - 1].isoformat(),
                "curr": dates[i].isoformat(),
                "delta_days": delta,
            })

    return dates


def _build_date_columns(day_tokens: List[Word], dates: List[Optional[date]]) -> Tuple[List[float], Dict[int, str]]:
    x_centers = [w.cx for w in day_tokens]
    col_to_date: Dict[int, str] = {}
    for idx, dt in enumerate(dates):
        if dt is not None:
            col_to_date[idx] = dt.isoformat()
    return x_centers, col_to_date


def _assign_to_nearest_column(x: float, x_centers: List[float]) -> Tuple[int, float]:
    best_i = 0
    best_d = float("inf")
    for i, cx in enumerate(x_centers):
        d = abs(x - cx)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def parse_schedule_pdf(
    pdf_bytes: bytes,
    year: Optional[int] = None,
    sort_output: bool = True,
    warn_on_ignored: bool = False,
    ignore_tokens: Optional[set] = None,
) -> Dict[str, object]:
    """Parse the PDF schedule into records + warnings + stats."""
    warnings: List[dict] = []
    ignore = set(DEFAULT_IGNORE_TOKENS) if ignore_tokens is None else set(ignore_tokens)

    words, full_text = _words_from_pdf_bytes(pdf_bytes)
    if not words:
        return {
            "records": [],
            "warnings": [{"type": "NO_TEXT", "message": "No text extracted from PDF."}],
            "stats": {"records": 0}
        }

    day_tokens, day_header_y = _detect_day_header(words)
    month_words = _detect_month_words(words, day_header_y)

    dates = _build_dates(day_tokens, month_words, full_text, year, warnings)
    x_centers, col_to_date = _build_date_columns(day_tokens, dates)

    if len(day_tokens) < 20:
        warnings.append({"type": "TOO_FEW_DATE_COLUMNS", "count": len(day_tokens)})

    date_left_edge = min(w.x0 for w in day_tokens)
    name_right_boundary = date_left_edge - 2.0

    body_words = [w for w in words if w.y0 > day_header_y + 5]
    rows = _cluster_by_y(body_words, tol=1.3)

    records: List[dict] = []
    unknown_codes = 0
    collisions = 0
    staff_rows = 0
    row_skipped = 0

    for row in rows:
        row_sorted = sorted(row, key=lambda w: w.x0)

        name_words = [w for w in row_sorted if w.x1 <= name_right_boundary]
        if not name_words:
            continue

        name_line = " ".join(w.text for w in name_words)
        name_line = re.sub(r"^\s*\d+\s+", "", name_line).strip()

        if "," not in name_line:
            row_skipped += 1
            continue

        staff_rows += 1
        staff_name = _normalize_name(name_line)

        date_region = [w for w in row_sorted if w.x0 >= date_left_edge - 1]
        if not date_region:
            continue

        cell_map: Dict[int, List[Word]] = {}
        for w in date_region:
            col_idx, dist = _assign_to_nearest_column(w.cx, x_centers)
            if dist > 4.5:
                continue
            cell_map.setdefault(col_idx, []).append(w)

        for col_idx, ws_in_cell in cell_map.items():
            if col_idx not in col_to_date:
                continue
            dt = col_to_date[col_idx]

            if len(ws_in_cell) > 1:
                collisions += 1
                warnings.append({
                    "type": "CELL_COLLISION",
                    "name": staff_name,
                    "date": dt,
                    "values": [w.text for w in sorted(ws_in_cell, key=lambda w: w.x0)],
                })

            w = sorted(ws_in_cell, key=lambda w: w.x0)[0]
            raw = _clean_text(w.text)
            key = raw.upper()

            if key in SHIFT_MAP:
                desc, status = SHIFT_MAP[key]
                records.append({
                    "date": dt,
                    "name": staff_name,
                    "shift_type": raw,
                    "description": desc,
                    "status": status,
                })
            else:
                if key in ignore:
                    if warn_on_ignored:
                        warnings.append({
                            "type": "IGNORED_TOKEN",
                            "name": staff_name,
                            "date": dt,
                            "value": raw,
                        })
                    continue

                unknown_codes += 1
                warnings.append({
                    "type": "UNKNOWN_SHIFT_CODE",
                    "name": staff_name,
                    "date": dt,
                    "value": raw,
                })

    if not records:
        warnings.append({"type": "NO_RECORDS", "message": "No valid shift records extracted."})

    if sort_output:
        records.sort(key=lambda r: (r["date"], r["name"], r["shift_type"]))

    stats = {
        "pages": len(set(w.page for w in words)),
        "words": len(words),
        "date_columns": len(day_tokens),
        "staff_rows_detected": staff_rows,
        "records": len(records),
        "unknown_shift_codes": unknown_codes,
        "cell_collisions": collisions,
        "rows_skipped_non_staff": row_skipped,
        "date_range": {
            "start": min((r["date"] for r in records), default=None),
            "end": max((r["date"] for r in records), default=None),
        },
    }

    return {"records": records, "warnings": warnings, "stats": stats}

# -------------------- FastAPI --------------------

app = FastAPI(title="PDF Nursing Schedule → JSON", version="1.1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/convert-pdf")
async def convert_pdf(
    file: UploadFile = File(...),
    year: Optional[int] = Form(default=None),
    envelope: bool = Form(default=True),
    sort_output: bool = Form(default=True),
    warn_on_ignored: bool = Form(default=False),
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Convert schedule PDF into JSON records.

    Required header:
    - x-api-key: must match API_KEY in Render environment variables

    Form fields:
    - file: uploaded PDF
    - year: optional int
    - envelope: if true returns {records,warnings,stats}, else returns records[]
    - sort_output: sort records chronologically
    - warn_on_ignored: emit warnings for ignored tokens like SS/IC/etc.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    pdf_bytes = await file.read()

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        result = parse_schedule_pdf(
            pdf_bytes,
            year=year,
            sort_output=sort_output,
            warn_on_ignored=warn_on_ignored,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

    if envelope:
        return JSONResponse(content=result)
    return JSONResponse(content=result["records"])
