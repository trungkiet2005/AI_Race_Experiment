# %%
"""Build a portable vLLM wheelhouse in a Kaggle notebook with Internet ON.

Use the same Kaggle image/Python version as the offline GPU notebook. Save the
``/kaggle/working/vllm_wheels`` output as a Kaggle Dataset, then add that Dataset
to the Internet-OFF experiment notebook.
"""

# %%
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

VLLM_SPEC = os.environ.get("VLLM_SPEC", "").strip()
if not VLLM_SPEC:
    raise RuntimeError(
        "Set VLLM_SPEC to an exact version, for example vllm==<audited-version>."
    )
EXTRA_SPECS = [
    spec.strip()
    for spec in os.environ.get("VLLM_EXTRA_SPECS", "").split(",")
    if spec.strip()
]
PACKAGE_SPECS = [VLLM_SPEC, *EXTRA_SPECS]
unpinned = [
    spec
    for spec in PACKAGE_SPECS
    if "==" not in spec
]
if unpinned:
    raise RuntimeError(
        f"Every wheelhouse requirement must be exactly pinned with ==: {unpinned}"
    )

OUTPUT_ROOT = (
    Path("/kaggle/working")
    if Path("/kaggle/working").is_dir()
    else Path.cwd()
)
WHEELS_DIR = OUTPUT_ROOT / "vllm_wheels"
if WHEELS_DIR.exists() and any(WHEELS_DIR.iterdir()):
    raise RuntimeError(
        f"{WHEELS_DIR} is not empty. Start from a fresh Kaggle session or choose "
        "a new output directory so wheels from different resolutions cannot mix."
    )
WHEELS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %%
# Binary-only is deliberate: an offline notebook should never need to compile an sdist.
download_command = [
    sys.executable,
    "-m",
    "pip",
    "download",
    "--dest",
    str(WHEELS_DIR),
    "--only-binary=:all:",
    *PACKAGE_SPECS,
]
print("Downloading:", " ".join(PACKAGE_SPECS))
subprocess.check_call(download_command)

wheel_files = sorted(WHEELS_DIR.glob("*.whl"))
if not wheel_files:
    raise RuntimeError("pip completed but produced no wheel files.")


# %%
# Resolve the wheelhouse without network access; this catches missing dependencies.
verify_command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--dry-run",
    "--no-index",
    f"--find-links={WHEELS_DIR}",
    *PACKAGE_SPECS,
]
subprocess.check_call(verify_command)

manifest = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "requested": PACKAGE_SPECS,
    "files": [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in wheel_files
    ],
    "total_bytes": sum(path.stat().st_size for path in wheel_files),
}
(WHEELS_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print(f"Wheelhouse ready: {WHEELS_DIR}")
print(f"{len(wheel_files)} wheels, {manifest['total_bytes'] / 1024**3:.2f} GiB")
print("Save this output directory as a Kaggle Dataset.")
