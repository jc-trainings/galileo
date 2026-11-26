"""Download CropHarvest reliably: resumes after dropped connections and checks sizes.

The repo's own downloader writes straight to disk without checking the size, so a dropped
connection leaves a truncated .tar.gz that fails to unpack ("Compressed file ended
before the end-of-stream marker was reached"). It also only runs when
data/cropharvest_data does not exist yet, so a rerun does not repair anything.

This module downloads the three files the Kenya scripts need:
    features.tar.gz       (~83 MB)  -> features/
    test_features.tar.gz  (~786 MB) -> test_features/
    labels.geojson        (~86 MB)
into data/cropharvest_data, resuming partial downloads (.part files), checking each
file's size against the server, and cleaning up half-unpacked folders.

The Level 2 and Level 3 scripts call it automatically. You can also run it on its own:
    python case_study\\cropharvest_download.py
"""

import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.cropharvest.config import DATASET_URL  # noqa: E402

ARCHIVES = ["features", "test_features"]  # each comes as <name>.tar.gz
FILES = ["labels.geojson"]
MAX_ATTEMPTS = 20
CHUNK = 64 * 1024  # small chunks: less is lost when a connection drops


def _url(base: str, name: str) -> str:
    return f"{base}/files/{name}?download=1"


def _remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    size = int(r.headers.get("Content-Length", 0))
    if size <= 0:  # some servers omit it on HEAD; ask for the first byte instead
        r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("Content-Length", 0))
        r.close()
    return size


def download(url: str, dest: Path) -> Path:
    """Download url to dest, resuming from dest.part; verify the final size."""
    part = dest.with_name(dest.name + ".part")
    total = _remote_size(url)
    if dest.exists():
        size = dest.stat().st_size
        if not total or size == total:
            return dest
        if size < total and not part.exists():
            # e.g. a truncated file left by the repo's downloader: resume from it
            print(f"{dest.name} is incomplete ({size / 1e6:.1f} of {total / 1e6:.1f} MB); "
                  "resuming")
            dest.replace(part)
        else:
            dest.unlink()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have == total:
            break
        if total and have > total:  # corrupt partial file: start again
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: restart
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as fh, tqdm(
                    total=total or None, initial=have, unit="B", unit_scale=True,
                    desc=f"{dest.name} (attempt {attempt})",
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        bar.update(len(chunk))
        except (requests.RequestException, ConnectionError, OSError) as e:
            print(f"  connection problem: {e.__class__.__name__}: {e}")
        have = part.stat().st_size if part.exists() else 0
        if total and have == total:
            break
        if not total and have > 0:
            break  # size unknown: accept what we got (the extract step will check it)
        wait = min(60, 5 * attempt)
        print(f"  got {have / 1e6:.1f} of {total / 1e6:.1f} MB; resuming in {wait}s...")
        time.sleep(wait)
    else:
        raise RuntimeError(f"Could not download {dest.name} after {MAX_ATTEMPTS} attempts. "
                           f"Download it in a browser from {url} into {dest.parent}")
    part.replace(dest)
    return dest


def _extract(tar_path: Path, root: Path, name: str) -> None:
    target = root / name
    tmp = root / f"{name}.extracting"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    print(f"Unpacking {tar_path.name}...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(tmp)
    # the archive contains a top-level <name>/ folder
    inner = tmp / name
    src = inner if inner.is_dir() else tmp
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    if tmp.exists():
        shutil.rmtree(tmp)
    tar_path.unlink()


def ensure_cropharvest(root: Path = REPO_ROOT / "data" / "cropharvest_data",
                       base_url: str = DATASET_URL) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    for name in ARCHIVES:
        folder, tar_path = root / name, root / f"{name}.tar.gz"
        # The repo deletes the .tar.gz only after a successful unpack, so a folder plus a
        # leftover .tar.gz means an interrupted unpack: start that folder again.
        if folder.exists() and tar_path.exists():
            print(f"Removing half-unpacked {folder.name}/")
            shutil.rmtree(folder)
        if folder.exists():
            continue
        for attempt in (1, 2):
            download(_url(base_url, tar_path.name), tar_path)
            try:
                _extract(tar_path, root, name)
                break
            except (EOFError, tarfile.TarError, OSError) as e:
                print(f"  {tar_path.name} is damaged ({e}); downloading again")
                tar_path.unlink(missing_ok=True)
                if attempt == 2:
                    raise

    for name in FILES:
        path = root / name
        if path.exists():
            tail = path.read_bytes()[-64:].strip()
            if tail.endswith(b"}"):  # complete GeoJSON: nothing to do (works offline)
                continue
            print(f"{name} looks truncated; resuming it")  # download() resumes or restarts
        download(_url(base_url, name), path)

    kenya = sorted((root / "test_features").glob("Kenya_maize_2020_*.h5"))
    if len(kenya) < 2:
        raise RuntimeError(f"Expected Kenya test files in {root / 'test_features'}, "
                           f"found {len(kenya)}. Delete that folder and run again.")
    print(f"CropHarvest ready in {root} ({len(kenya)} Kenya test files)")
    return root


if __name__ == "__main__":
    ensure_cropharvest()
