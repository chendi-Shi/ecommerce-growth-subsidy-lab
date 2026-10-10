"""Acquire one workbook in non-overlapping, integrity-checked HTTP byte ranges.

Useful when an official or attributed mirror throttles a single large response.
No sample/subset is created. All bytes of one ETag-identified resource are joined.
The data license/source attribution remains UCI Online Retail II (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def ranged_download(url: str, output: Path, parts: int = 8) -> dict:
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
    if not etag or size <= 0:
        raise ValueError("Range acquisition requires a positive length and immutable ETag")
    output.parent.mkdir(parents=True, exist_ok=True)
    block = (size + parts - 1) // parts

    def fetch(index: int) -> Path:
        start, end = index * block, min((index + 1) * block, size) - 1
        target = output.with_suffix(output.suffix + f".part{index}")
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{end}", "If-Match": etag,
        })
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if (response.status != 206 or response.headers.get("Content-Range") != expected_range
                            or response.headers.get("ETag") != etag):
                        raise ValueError(f"Range response changed or was ignored: {index}")
                    with target.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 128 * 1024)
                if target.stat().st_size != end - start + 1:
                    raise ValueError(f"Range {index} length mismatch")
                print(f"Verified range {index + 1}/{parts}: {start}-{end}", flush=True)
                return target
            except (OSError, ValueError):
                if attempt == 2:
                    raise
        raise AssertionError("Unreachable")

    with ThreadPoolExecutor(max_workers=parts) as pool:
        files = {future: index for index, future in
                 ((i, pool.submit(fetch, i)) for i in range(parts))}
        ordered = {}
        for future in as_completed(files):
            ordered[files[future]] = future.result()
    partial = output.with_suffix(output.suffix + ".assembling")
    with partial.open("wb") as stream:
        for index in range(parts):
            with ordered[index].open("rb") as source:
                shutil.copyfileobj(source, stream, 1024 * 1024)
    if partial.stat().st_size != size:
        raise ValueError("Assembled workbook length mismatch")
    with zipfile.ZipFile(partial) as workbook:
        if workbook.testzip() is not None:
            raise ValueError("Assembled XLSX ZIP CRC check failed")
    partial.replace(output)
    for path in ordered.values():
        path.unlink()
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"download": url, "bytes": size, "etag": etag, "sha256": digest,
            "method": f"{parts} non-overlapping verified HTTP ranges; XLSX ZIP CRC checked"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/online_retail_II.xlsx"))
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.parts <= 16:
        parser.error("parts must be between 1 and 16")
    metadata = ranged_download(args.url, args.output, args.parts)
    args.output.with_suffix(".transport.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
