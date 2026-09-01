"""Model (.pth) and FAISS index (.index) locations.

Canonical layout:
  assets/weights/<model>.pth
  assets/indices/<speaker>.index

A speaker family maps every checkpoint of that voice onto one index:
  thchs_v2_e200_s13200.pth  ->  thchs_v2.index
  shanxi_e200_s11800.pth    ->  shanxi.index
  myvoice.pth               ->  myvoice.index
"""
from pathlib import Path

WEIGHTS_REL = Path("assets") / "weights"
INDICES_REL = Path("assets") / "indices"

# FAISS / GPU cosine retrieval. Realtime and offline conversion share this k.
INDEX_TOPK = 4

# Longest family first so thchs_female is not swallowed by a shorter prefix.
SPEAKER_FAMILIES = (
    "thchs_female",
    "thchs_v2",
    "myvoice",
    "shanxi",
)

INDEX_ALIASES = {
    "added_ivf2716_flat_nprobe_1_thchs_v2_v2.index": "thchs_v2.index",
    "added_ivf314_flat_nprobe_1_myvoice_v2.index": "myvoice.index",
}


def _name(path):
    return Path(str(path or "")).name


def speaker_family(path):
    """Return the speaker family stem, or '' if unknown."""
    stem = Path(_name(path)).stem.lower()
    if not stem:
        return ""
    for fam in SPEAKER_FAMILIES:
        if stem == fam or stem.startswith(fam + "_") or stem.startswith(fam + "-"):
            return fam
    return ""


def canonical_index_name(path):
    """Map a model or legacy FAISS filename to <family>.index."""
    name = _name(path)
    if not name:
        return ""
    alias = INDEX_ALIASES.get(name.lower())
    if alias:
        return alias
    fam = speaker_family(name)
    if fam:
        return fam + ".index"
    if name.lower().endswith(".index"):
        return name
    stem = Path(name).stem
    return (stem + ".index") if stem else ""


def _existing(path):
    try:
        p = Path(path)
        if p.is_file():
            return str(p.resolve())
    except Exception:
        return ""
    return ""


def _search_roots(project_root):
    root = Path(project_root)
    roots = [root]
    source = root / "source"
    if source.is_dir():
        roots.append(source)
    return roots


def resolve_model_path(path, project_root):
    if not path:
        return ""
    found = _existing(path)
    if found:
        return found
    name = _name(path)
    raw = Path(str(path))
    for root in _search_roots(project_root):
        for cand in (
            root / raw,
            root / WEIGHTS_REL / name,
        ):
            found = _existing(cand)
            if found:
                return found
    return str(path)


def resolve_index_path(path, project_root, model_path=""):
    """Resolve an index path; if missing, fall back to the model's family index."""
    names = []
    raw_name = _name(path)
    if raw_name:
        names.append(raw_name)
        canon = canonical_index_name(raw_name)
        if canon and canon not in names:
            names.append(canon)
    if model_path:
        fam = canonical_index_name(model_path)
        if fam and fam not in names:
            names.append(fam)
    if not names:
        return ""

    found = _existing(path) if path else ""
    if found:
        return found

    raw = Path(str(path)) if path else None
    for root in _search_roots(project_root):
        for name in names:
            family = Path(name).stem
            candidates = [
                root / INDICES_REL / name,
                root / "logs" / family / name,
            ]
            if raw is not None:
                candidates.insert(0, root / raw)
            for cand in candidates:
                found = _existing(cand)
                if found:
                    return found
    return str(path) if path else ""


def list_index_names(project_root):
    """Basenames of indexes available for inference (canonical dir first)."""
    seen = set()
    names = []
    for root in _search_roots(project_root):
        for folder in (root / INDICES_REL,):
            if not folder.is_dir():
                continue
            for p in sorted(folder.glob("*.index")):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    names.append(p.name)
        logs = root / "logs"
        if not logs.is_dir():
            continue
        for p in sorted(logs.rglob("*.index")):
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                names.append(p.name)
    return names
