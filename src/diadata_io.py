"""Shared, READ-ONLY helpers for loading DiaData.

Nothing in this file writes anywhere. Every script opens the raw CSVs only
for reading, so the files in data/raw/ are never changed.

Column names come from the DiaData Zenodo/openHSU description:
  Main database:  ts, PtID, GlucoseCGM, Database
  Demographics:   PtID plus Age / AgeGroup / AgeGroupBroad / Sex (depends on version)
[VERIFY] Open the first lines of your CSVs and confirm these names. If they
differ, change the constants below - nothing else needs editing.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---- column names (edit here if your DiaData version differs) --------------
TS, PID, GLU, DB = "ts", "PtID", "GlucoseCGM", "Database"
AGE, AGEGROUP, AGEGROUP_BROAD = "Age", "AgeGroup", "AgeGroupBroad"

# ---- PRIMARY analysis settings (one place; robustness runs override by flags) ----
# Low = glucose < 70 mg/dL for >= 4 consecutive 5-min readings (>= 15 min between
# first and last reading), night 00:00-06:00, as in Eichenlaub et al., J Diabetes
# Sci Technol 2024;18(5):1035-1043 (doi 10.1177/19322968241267765) and the
# international CGM consensus (<70 mg/dL, >= 15 min).
PRIMARY = {"night_start": 0, "night_end": 6, "threshold": 70.0, "op": "lt", "min_run": 4}


def make_tag(night_start, night_end, op, threshold, min_run) -> str:
    return f"n{night_start:02d}-{night_end:02d}_{op}{threshold:g}_r{min_run}"


PRIMARY_TAG = make_tag(PRIMARY["night_start"], PRIMARY["night_end"], PRIMARY["op"],
                       PRIMARY["threshold"], PRIMARY["min_run"])

SLOT_MIN = 5                      # DiaData is on a 5-minute grid
SLOTS_PER_HOUR = 60 // SLOT_MIN   # 12
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR  # 288
_EPOCH = pd.Timestamp("1970-01-01")

# Age-group codes as documented for DiaData v3
AGEGROUP_FINE = {0: "0-2", 1: "3-6", 2: "7-10", 3: "11-13", 4: "14-17",
                 5: "18-25", 6: "26-35", 7: "36-55", 8: "56-100"}
AGEGROUP_BROAD_LABELS = {0: "0-13", 1: "14-24", 2: "25-44", 3: "45-100"}
# Children / teens / adults built from the FINE groups (no group straddles a cut)
FINE_TO_3 = {0: "child 0-13", 1: "child 0-13", 2: "child 0-13", 3: "child 0-13",
             4: "teen 14-17", 5: "adult 18+", 6: "adult 18+", 7: "adult 18+",
             8: "adult 18+"}
AGE_SCHEMES = ("broad", "kids_teens_adults", "fine")
UNKNOWN = "unknown"


def sha256(path: Path, block: int = 1 << 20) -> str:
    """Checksum of a file, so you can prove later the raw data never changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:           # "rb" = read-only
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def parse_ts(s: pd.Series) -> pd.Series:
    try:
        out = pd.to_datetime(s, format="ISO8601")
    except (ValueError, TypeError):
        out = pd.to_datetime(s, format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        warnings.warn(
            "Timestamps carry a time-zone offset (e.g. UTC). Clock-based nights "
            "assume LOCAL time - check this before trusting any night results. "
            "The offset is being dropped as-is.", stacklevel=2)
        out = out.dt.tz_localize(None)
    return out


def to_slot(ts: pd.Series) -> np.ndarray:
    """Timestamp -> integer 5-min slot number since 1970 (rounded to nearest)."""
    minutes = (ts - _EPOCH) / pd.Timedelta(minutes=1)
    return np.round(minutes.to_numpy(dtype="float64") / SLOT_MIN).astype("int64")


def load_main_arrays(path: Path, chunksize: int = 2_000_000):
    """Read the main database into compact numpy arrays, sorted by subject then time.

    Returns dict with:
      code  int32   subject index (0..n-1)
      slot  int32   5-min slot since 1970
      glu   float32 glucose mg/dL (NaN if missing)
      pids  list    subject IDs (pids[code])
      db    list    source dataset for each subject (most common value)
      tz_aware bool
    """
    cols = read_header(path)
    need = {TS, PID, GLU}
    if need - set(cols):
        raise SystemExit(f"Main CSV is missing columns {need - set(cols)}. "
                         f"Found: {cols}. Edit the constants in diadata_io.py.")
    usecols = [c for c in (TS, PID, GLU, DB) if c in cols]

    pid_to_code: dict[str, int] = {}
    db_counts: dict[int, dict[str, int]] = {}
    codes, slots, glus = [], [], []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype={PID: str, DB: str}):
        ts = parse_ts(chunk[TS])
        ok = ts.notna().to_numpy()
        local_codes, uniques = pd.factorize(chunk[PID].astype(str))
        mapping = np.array([pid_to_code.setdefault(u, len(pid_to_code)) for u in uniques],
                           dtype="int32")
        c = mapping[local_codes]
        dbs = chunk[DB].astype(str).to_numpy() if DB in chunk else np.full(len(chunk), UNKNOWN)
        tmp = pd.DataFrame({"c": c, "d": dbs}).value_counts()
        for (code_val, db_val), n in tmp.items():
            db_counts.setdefault(int(code_val), {}).setdefault(db_val, 0)
            db_counts[int(code_val)][db_val] += int(n)
        codes.append(c[ok])
        slots.append(to_slot(ts[ok]).astype(np.int32))   # 5-min slots since 1970 fit in int32 (until ~2290)
        glus.append(pd.to_numeric(chunk[GLU], errors="coerce").to_numpy("float32")[ok])

    code = np.concatenate(codes)
    slot = np.concatenate(slots)
    glu = np.concatenate(glus)
    order = np.lexsort((slot, code))          # sort by subject, then time
    pids = [None] * len(pid_to_code)
    for p, i in pid_to_code.items():
        pids[i] = p
    db = [max(db_counts.get(i, {UNKNOWN: 1}).items(), key=lambda kv: kv[1])[0]
          for i in range(len(pids))]
    return {"code": code[order], "slot": slot[order], "glu": glu[order],
            "pids": pids, "db": db}


def split_by_subject(code: np.ndarray):
    """Yield (subject_code, start, stop) index ranges into the sorted arrays."""
    if len(code) == 0:
        return
    breaks = np.flatnonzero(np.diff(code)) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(code)]
    for a, b in zip(starts, stops):
        yield int(code[a]), int(a), int(b)


def collapse_duplicate_slots(slot: np.ndarray, glu: np.ndarray):
    """If two readings fall in the same 5-min slot, keep their mean.
    Returns unique slots, mean glucose (NaN if all missing), n rows collapsed."""
    u, inv = np.unique(slot, return_inverse=True)
    present = ~np.isnan(glu)
    sums = np.bincount(inv, weights=np.where(present, glu, 0.0), minlength=len(u))
    cnts = np.bincount(inv, weights=present.astype(float), minlength=len(u))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts > 0, sums / cnts, np.nan).astype("float32")
    return u, mean, int(len(slot) - len(u))


# ---- demographics -----------------------------------------------------------
# Age bins (lower bound inclusive, in whole years) used when the file has an Age
# column but no AgeGroup codes (DiaData v3 SDBI). Labels match the code tables above.
AGE_BINS = {
    "broad": [(0, 13, "0-13"), (14, 24, "14-24"), (25, 44, "25-44"), (45, 200, "45-100")],
    "kids_teens_adults": [(0, 13, "child 0-13"), (14, 17, "teen 14-17"), (18, 200, "adult 18+")],
    "fine": [(0, 2, "0-2"), (3, 6, "3-6"), (7, 10, "7-10"), (11, 13, "11-13"), (14, 17, "14-17"),
             (18, 25, "18-25"), (26, 35, "26-35"), (36, 55, "36-55"), (56, 200, "56-100")],
}


def _parse_age(v: str):
    """'12' / '12.0' -> (12.0, 12.0); '60-100' -> (60.0, 100.0); else None."""
    s = str(v).strip()
    try:
        x = float(s)
        return (x, x)
    except ValueError:
        pass
    if "-" in s:
        a, _, b = s.partition("-")
        try:
            return (float(a), float(b))
        except ValueError:
            return None
    return None


def load_demographics(path: Path | None, chunksize: int = 2_000_000) -> pd.DataFrame:
    """One row per subject.

    Two layouts are supported:
      * AgeGroup / AgeGroupBroad code columns (older DiaData descriptions), or
      * DiaData v3 Subdatabase I (SDBI): ts, PtID, GlucoseCGM, Age, Sex, Database,
        where Age is the subject's age AT EACH READING (it increases during
        multi-year studies) and a few sources give only a range (e.g. '60-100').
    For the v3 layout each subject gets ONE age: the age at their first reading
    (= smallest numeric age), so a child who turns 14 during the study stays in
    the age group they were in at enrollment. Range-only subjects keep the range.
    Columns returned: Age (representative value as text), age_low, age_high,
    n_distinct_age_values, plus AgeGroup/AgeGroupBroad if present.
    """
    if path is None:
        return pd.DataFrame(columns=[PID]).set_index(PID)
    cols = read_header(path)
    if PID not in cols:
        raise SystemExit(f"Demographics CSV has no '{PID}' column. Found: {cols}")
    keep = [c for c in (PID, AGE, AGEGROUP, AGEGROUP_BROAD) if c in cols]
    if len(keep) == 1:
        raise SystemExit(f"Demographics CSV has no Age/AgeGroup column. Found: {cols}")
    parts = [ch.drop_duplicates() for ch in
             pd.read_csv(path, usecols=keep, chunksize=chunksize, dtype=str)]
    demo = pd.concat(parts).drop_duplicates()
    demo[PID] = demo[PID].astype(str)

    out = pd.DataFrame(index=pd.Index(sorted(demo[PID].unique()), name=PID))
    for c in (AGEGROUP, AGEGROUP_BROAD):
        if c in demo:
            out[c] = demo.dropna(subset=[c]).groupby(PID)[c].first()
    if AGE in demo:
        a = demo[[PID, AGE]].dropna()
        parsed = a[AGE].map(_parse_age)
        a = a.assign(lo=parsed.map(lambda t: t[0] if t else np.nan),
                     hi=parsed.map(lambda t: t[1] if t else np.nan),
                     is_num=parsed.map(lambda t: bool(t) and t[0] == t[1]))
        num = a[a["is_num"]].groupby(PID)["lo"].min()          # age at first reading
        rng = a[~a["is_num"] & a["lo"].notna()].groupby(PID).agg(lo=("lo", "min"),
                                                                   hi=("hi", "max"))
        out["age_low"] = num.reindex(out.index)
        out["age_high"] = num.reindex(out.index)
        only_rng = out["age_low"].isna() & out.index.isin(rng.index)
        out.loc[only_rng, "age_low"] = rng["lo"].reindex(out.index)[only_rng]
        out.loc[only_rng, "age_high"] = rng["hi"].reindex(out.index)[only_rng]
        rep = num.map(lambda x: f"{x:g}").reindex(out.index)
        rep[only_rng] = [f"{lo:g}-{hi:g}" for lo, hi in
                         zip(out.loc[only_rng, "age_low"], out.loc[only_rng, "age_high"])]
        out[AGE] = rep
        out["n_distinct_age_values"] = a.groupby(PID)[AGE].nunique().reindex(out.index).fillna(0).astype(int)
    return out


def age_type(value) -> str:
    """'exact' if Age is a single number, 'range' if it is e.g. '60-100',
    'missing' if empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "na"}:
        return "missing"
    try:
        float(s)
        return "exact"
    except ValueError:
        return "range"


def _bin_label(lo: float, hi: float, bins) -> str:
    """Group label if the whole age (or age range) falls inside one bin; whole
    years only (an age of 13.9 counts as 13)."""
    if np.isnan(lo):
        return UNKNOWN
    lo_y, hi_y = int(np.floor(lo)), int(np.floor(hi))
    for a, b, lab in bins:
        if a <= lo_y <= b and a <= hi_y <= b:
            return lab
    return UNKNOWN          # a range that straddles two bins cannot be assigned


def age_labels(demo: pd.DataFrame, scheme: str) -> pd.Series:
    """Map each subject to an age-group label under the chosen scheme."""
    if scheme not in AGE_SCHEMES:
        raise SystemExit(f"--age-scheme must be one of {AGE_SCHEMES}")
    if demo.empty:
        return pd.Series(dtype=str)
    if "age_low" in demo.columns:                        # DiaData v3 layout
        bins = AGE_BINS[scheme]
        return pd.Series([_bin_label(lo, hi, bins) for lo, hi in
                          zip(demo["age_low"], demo["age_high"])], index=demo.index)
    col, mapping = {
        "broad": (AGEGROUP_BROAD, AGEGROUP_BROAD_LABELS),
        "kids_teens_adults": (AGEGROUP, FINE_TO_3),
        "fine": (AGEGROUP, AGEGROUP_FINE),
    }[scheme]
    if col not in demo.columns:
        raise SystemExit(f"Scheme '{scheme}' needs column '{col}', which is not in "
                         f"the demographics file (has {list(demo.columns)}).")
    codes = pd.to_numeric(demo[col], errors="coerce").astype("Int64")
    return codes.map(mapping).fillna(UNKNOWN).astype(str)


def check_output_dir(out: Path, raw_paths: list[Path]) -> None:
    """Refuse to write outputs into the folder that holds the raw files."""
    out = out.resolve()
    for p in raw_paths:
        if p is not None and out == p.resolve().parent:
            raise SystemExit(f"Output folder {out} is the raw-data folder. "
                             "Choose results/ or data/processed/ instead.")
