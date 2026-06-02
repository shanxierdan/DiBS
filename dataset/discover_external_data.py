#!/usr/bin/env python3
"""Discover and validate external dataset sources for Table4 extension.

Strict admission policy:
1. explicit license
2. reproducible download/reference
3. parseable format
4. optional availability check
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "external_sources_manifest.md"
DEFAULT_ACCEPTED = PROJECT_ROOT / "dataset" / "accepted_sources.json"
DEFAULT_REJECTED = PROJECT_ROOT / "dataset" / "rejected_sources.json"

ALLOWED_LICENSE_PREFIXES = {
    "mit",
    "apache",
    "bsd",
    "cc-by",
    "cc0",
    "odc",
    "gpl",
    "lgpl",
    "mpl",
    "local-project-data",
}


def _extract_json_block(text: str) -> List[Dict]:
    start = text.find("```json")
    if start < 0:
        raise ValueError("No JSON block found in manifest.")
    start = text.find("\n", start)
    end = text.find("```", start + 1)
    if start < 0 or end < 0:
        raise ValueError("Malformed JSON block in manifest.")
    payload = text[start:end].strip()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Manifest JSON payload must be a list.")
    return data


def _license_ok(license_name: str) -> bool:
    value = (license_name or "").strip().lower()
    return any(value.startswith(prefix) for prefix in ALLOWED_LICENSE_PREFIXES)


def _has_reproducible_download(source: Dict) -> bool:
    download = (source.get("download") or "").strip().lower()
    url = (source.get("url") or "").strip()
    return bool(download) and (
        download == "already-in-repo" or url.startswith("http://") or url.startswith("https://") or url.startswith("file://")
    )


def _format_supported(task_family: str, fmt: str) -> bool:
    fmt = (fmt or "").lower()
    if task_family == "generalized_sudoku":
        return "puzzle" in fmt or "text" in fmt or "csv" in fmt or "json" in fmt
    if task_family == "nqueens":
        return "puzzle" in fmt or "text" in fmt or "json" in fmt
    return False


def _check_url_availability(url: str, timeout: float = 5.0) -> Tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        if parsed.netloc:
            local_path = PROJECT_ROOT / parsed.netloc / parsed.path.lstrip("/")
        elif os.path.isabs(parsed.path):
            local_path = Path(parsed.path)
        else:
            local_path = PROJECT_ROOT / parsed.path
        if local_path.exists():
            return True, "local file exists"
        return False, f"local file missing: {local_path}"
    if parsed.scheme in ("http", "https"):
        try:
            req = Request(url, method="HEAD")
            with urlopen(req, timeout=timeout) as resp:
                code = getattr(resp, "status", 200)
            if 200 <= code < 400:
                return True, f"http status {code}"
            return False, f"http status {code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"http check failed: {exc}"
    return False, f"unsupported URL scheme: {parsed.scheme or 'none'}"


def evaluate_source(source: Dict, check_network: bool) -> Tuple[bool, str]:
    required_keys = ("source_id", "task_family", "size", "url", "license", "download", "format")
    for key in required_keys:
        if not source.get(key):
            return False, f"missing required field: {key}"

    if source["task_family"] not in ("generalized_sudoku", "nqueens"):
        return False, f"unsupported task_family: {source['task_family']}"
    if not _license_ok(source["license"]):
        return False, f"license not accepted: {source['license']}"
    if not _has_reproducible_download(source):
        return False, "download field is not reproducible"
    if not _format_supported(source["task_family"], source["format"]):
        return False, f"unsupported format notes: {source['format']}"

    if check_network:
        ok, reason = _check_url_availability(source["url"])
        if not ok:
            return False, reason
    return True, "accepted"


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and validate external data sources.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--accepted-out", type=str, default=str(DEFAULT_ACCEPTED))
    parser.add_argument("--rejected-out", type=str, default=str(DEFAULT_REJECTED))
    parser.add_argument("--check-network", action="store_true", help="Run URL/file availability checks.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_text = manifest_path.read_text(encoding="utf-8")
    candidates = _extract_json_block(manifest_text)

    accepted: List[Dict] = []
    rejected: List[Dict] = []

    for source in candidates:
        ok, reason = evaluate_source(source, check_network=args.check_network)
        row = dict(source)
        row["validation"] = reason
        row["status"] = "accepted" if ok else "rejected"
        if ok:
            accepted.append(row)
        else:
            rejected.append(row)

    Path(args.accepted_out).write_text(json.dumps(accepted, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.rejected_out).write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Candidates: {len(candidates)}")
    print(f"Accepted : {len(accepted)} -> {args.accepted_out}")
    print(f"Rejected : {len(rejected)} -> {args.rejected_out}")


if __name__ == "__main__":
    main()
