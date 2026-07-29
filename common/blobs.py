"""Blob storage: gzipped JSON on local disk, keyed by relative path.

Dev = local dir; production swaps BLOB_DIR for a mounted volume or S3 sync target.
"""
import gzip
import json
from pathlib import Path

from common.config import BLOB_DIR


def put_json(key: str, obj) -> str:
    path = Path(BLOB_DIR) / key
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    return key


def get_json(key: str):
    path = Path(BLOB_DIR) / key
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def delete(key: str) -> None:
    path = Path(BLOB_DIR) / key
    path.unlink(missing_ok=True)
