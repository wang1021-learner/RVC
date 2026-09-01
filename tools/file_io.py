import json
import os
import tempfile


def read_text(path, errors="strict", newline=None):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=encoding, errors="strict", newline=newline) as file:
                return file.read()
        except UnicodeDecodeError as error:
            last_error = error
    if errors != "strict":
        with open(path, "r", encoding="gbk", errors=errors, newline=newline) as file:
            return file.read()
    raise last_error


def write_json_atomic(path, data, encoding="utf-8"):
    """先写临时文件再替换，避免写到一半断电把配置写坏。"""
    path = os.fspath(path)
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
