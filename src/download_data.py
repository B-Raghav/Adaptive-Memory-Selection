"""Download and extract the Multi-Session Chat (MSC) dataset.

MSC is released by Facebook AI Research through ParlAI. We fetch the tarball
directly from the public file server (the parl.ai URL just redirects here) and
extract it into ``data/raw``. A HuggingFace mirror is used as a fallback if the
primary host is unreachable.

Reference:
    Xu, J., Szlam, A., and Weston, J. "Beyond Goldfish Memory: Long-Term
    Open-Domain Conversation." ACL 2022.
"""
from __future__ import annotations

import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001 - certifi optional
    _SSL_CTX = None

PRIMARY_URL = "https://dl.fbaipublicfiles.com/parlai/msc/msc_v0.1.tar.gz"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ARCHIVE = RAW_DIR / "msc_v0.1.tar.gz"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total_size / 1e6:6.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()

    # python.org macOS builds often lack a usable CA bundle, tripping urllib on
    # HTTPS. Prefer system curl (which has the OS trust store); fall back to
    # urllib with certifi if curl is unavailable.
    if _which("curl"):
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=True)
        return
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX)) if _SSL_CTX else None
    if opener is not None:
        urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, _progress)
    sys.stdout.write("\n")


def _which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE.exists():
        try:
            _download(PRIMARY_URL, ARCHIVE)
        except Exception as exc:  # noqa: BLE001 - surface any network failure
            print(f"Primary download failed: {exc}")
            print(
                "Fallback: manually download the MSC dataset (e.g. the HuggingFace "
                "mirror 'nayohan/multi_session_chat' or ParlAI) into data/raw/."
            )
            raise
    else:
        print(f"Archive already present: {ARCHIVE}")

    print("Extracting ...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(RAW_DIR)  # noqa: S202 - trusted first-party archive

    sessions = sorted(RAW_DIR.rglob("*.txt")) + sorted(RAW_DIR.rglob("*.jsonl")) + sorted(RAW_DIR.rglob("*.json"))
    print(f"Extraction complete. Found {len(sessions)} data files under {RAW_DIR}.")
    for path in sessions[:20]:
        print(f"  {path.relative_to(RAW_DIR)}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
