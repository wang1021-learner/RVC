"""Model (.pth) and FAISS index (.index) locations.

Canonical layout:
  assets/weights/<model>.pth
  assets/indices/<speaker>.index

A speaker family maps every checkpoint of that voice onto one index:
  shanxi_e200_s11800.pth    ->  shanxi.index
  myvoice.pth               ->  myvoice.index
"""
import shutil
from pathlib import Path

WEIGHTS_REL = Path("assets") / "weights"
INDICES_REL = Path("assets") / "indices"

# FAISS / GPU cosine retrieval. Realtime and offline conversion share this k.
INDEX_TOPK = 4

SPEAKER_FAMILIES = (
    "myvoice",
    "shanxi",
)

INDEX_ALIASES = {
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


def writable_asset_dir(kind="weights"):
    """Directory for user-imported models/indexes (survives exe upgrades)."""
    from tools.app_paths import is_frozen, package_root

    rel = WEIGHTS_REL if kind != "indices" else INDICES_REL
    pkg = package_root()
    if is_frozen():
        source = pkg / "source"
        base = source if source.is_dir() else pkg
    else:
        base = pkg
    dest = base / rel
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def import_user_asset(src, kind="weights", overwrite=False):
    """Copy a .pth/.index into the local assets folder. Returns the basename."""
    src_p = Path(str(src or "")).expanduser()
    if not src_p.is_file():
        raise FileNotFoundError(str(src_p))
    dest = writable_asset_dir(kind) / src_p.name
    try:
        if dest.resolve() == src_p.resolve():
            return dest.name
    except Exception:
        pass
    if dest.is_file() and not overwrite:
        raise FileExistsError(dest.name)
    shutil.copy2(src_p, dest)
    return dest.name


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
        # Server copies often prefix the speaker id: OP2694892_added_IVF....index
        folder = root / INDICES_REL
        if folder.is_dir():
            for name in names:
                if not name:
                    continue
                for p in sorted(folder.glob("*" + name)):
                    found = _existing(p)
                    if found:
                        return found
        logs = root / "logs"
        if logs.is_dir():
            for name in names:
                if not name:
                    continue
                for p in logs.rglob(name):
                    found = _existing(p)
                    if found:
                        return found
    return str(path) if path else ""


def list_asset_names(kind="weights", project_root=None):
    """Basenames already on disk (.pth or .index)."""
    from tools.app_paths import package_root

    ext = "*.pth" if kind != "indices" else "*.index"
    rel = WEIGHTS_REL if kind != "indices" else INDICES_REL
    root = Path(project_root) if project_root else package_root()
    seen = set()
    names = []
    folders = []
    for base in _search_roots(root):
        folders.append(base / rel)
    try:
        wd = writable_asset_dir(kind)
        if wd not in folders:
            folders.insert(0, wd)
    except Exception:
        pass
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob(ext)):
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                names.append(p.name)
    return names


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
