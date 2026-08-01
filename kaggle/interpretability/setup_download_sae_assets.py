# %%
"""Build an offline wheelhouse + SAE weight cache for the Qwen2.5 SAE probe.

Run this in a Kaggle notebook with **Internet ON** (no GPU needed — this step
only downloads files). Save the output directory as a Kaggle Dataset, then add
that Dataset to ``qwen25_sae_probe.py``'s Internet-OFF run so it can attach to
the same competition-gated GPU machine shape as this project's other
notebooks (``kaggle/kernel-metadata.json`` uses ``competition_sources`` for
GPU access, which typically requires ``enable_internet: false``).

Mirrors ``kaggle/setup/build_quant_wheels.py``'s pattern (pinned versions,
binary-only wheels, a dry-run install to catch missing dependencies before
saving, and a SHA-256 manifest) but adds a second stage: downloading the
pretrained SAE weight files for the exact layer(s)
``qwen25_sae_probe.py`` will load, since ``SAE.from_pretrained`` normally
hits the Hugging Face Hub directly and that path is unavailable with
Internet OFF.

Quy trình (see kaggle/interpretability/README.md for the full context):

  1. Copy this file into a Kaggle notebook, Internet ON, GPU not required.
  2. Run All. Output: /kaggle/working/sae_probe_assets/{wheels/, sae/, manifest.json}
  3. Save that output directory as a Kaggle Dataset.
  4. Add the Dataset to the probe notebook; qwen25_sae_probe.py auto-detects
     it (same ``find_wheels_dir``-style search used by this project's other
     offline-install notebooks) and skips the live pip install / HF download.
"""

# %%
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

# Pinned exact versions -- update deliberately, not "latest", same rule as
# kaggle/setup/build_quant_wheels.py. Re-verify LAYER_IDS_TO_CACHE below still
# matches qwen25_sae_probe.py's SAE_RELEASE/LAYER before re-running.
SAE_LENS_SPEC = "sae-lens==6.47.0"
TRANSFORMER_LENS_SPEC = "transformer-lens==3.6.0"
PACKAGE_SPECS = [SAE_LENS_SPEC, TRANSFORMER_LENS_SPEC]

SAE_RELEASE_REPO_ID = "andyrdt/saes-qwen2.5-7b-instruct"
# One folder per layer in the HF repo; keep this in sync with the LAYER default
# (and any alternates you want cached) in qwen25_sae_probe.py.
LAYER_IDS_TO_CACHE = [15]
SAE_FOLDER_TEMPLATE = "resid_post_layer_{layer}/trainer_1"

OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
ASSETS_DIR = OUTPUT_ROOT / "sae_probe_assets"
WHEELS_DIR = ASSETS_DIR / "wheels"
SAE_DIR = ASSETS_DIR / "sae"
if ASSETS_DIR.exists() and any(ASSETS_DIR.iterdir()):
    raise RuntimeError(
        f"{ASSETS_DIR} is not empty. Start from a fresh Kaggle session so wheels/SAE "
        "files from different resolutions cannot mix into one manifest."
    )
WHEELS_DIR.mkdir(parents=True, exist_ok=True)
SAE_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %%
# Stage 1: download sae-lens + transformer-lens and their dependency closure.
#
# Use `pip wheel`, not `pip download`: at least one transitive dependency
# (transformers-stream-generator, pulled in by transformer-lens for legacy
# Qwen1 "QWenLMHeadModel" loading -- unused by this probe's Qwen2.5 model,
# but still a hard install_requires) only ships an sdist, and building an
# sdist needs a PEP 517 build backend. `pip download` defers that build to
# install time; verified in practice that the later Internet-OFF
# `pip install --no-index --find-links=...` build-isolation step can fail
# there in a way it does not when building right now, with Internet still
# available. `pip wheel` builds every sdist into a real .whl immediately, so
# the offline install in qwen25_sae_probe.py never needs to build anything.
build_command = [
    sys.executable, "-m", "pip", "wheel",
    "--wheel-dir", str(WHEELS_DIR),
    *PACKAGE_SPECS,
]
print("Building wheelhouse (download + build any sdists now):", " ".join(PACKAGE_SPECS))
subprocess.check_call(build_command)

wheel_files = sorted(WHEELS_DIR.glob("*.whl"))
sdist_files = sorted(
    p for p in WHEELS_DIR.iterdir() if p.suffix in (".gz", ".zip") and p.name.endswith((".tar.gz", ".zip"))
)
if not wheel_files:
    raise RuntimeError("pip wheel completed but produced no .whl files.")
if sdist_files:
    # Should not happen with `pip wheel` (unlike `pip download`, it builds
    # everything into .whl); fail loudly rather than ship an unbuildable sdist.
    raise RuntimeError(
        f"pip wheel left {len(sdist_files)} unbuilt sdist(s) in {WHEELS_DIR}: "
        f"{[p.name for p in sdist_files]} -- investigate before trusting this wheelhouse."
    )

# Resolve without network access -- catches anything pip silently expected to
# already be satisfied by the *current* (Internet-ON) environment but that
# the offline GPU notebook's environment might not actually have.
verify_command = [
    sys.executable, "-m", "pip", "install", "--dry-run",
    "--no-index", f"--find-links={WHEELS_DIR}",
    *PACKAGE_SPECS,
]
subprocess.check_call(verify_command)
print(f"wheelhouse OK: {len(wheel_files)} wheels + {len(sdist_files)} sdists")

# %%
# Stage 2: download the pretrained SAE weight files for the target layer(s).
#
# Deliberately populated as a STANDARD huggingface_hub cache (HF_HOME=SAE_DIR,
# no local_dir= override) rather than a custom layout: SAELens' loader calls
# plain hf_hub_download(repo_id=..., filename=...) internally with no local_dir
# either, so on the Internet-OFF probe notebook, pointing HF_HOME at this same
# cache directory (see qwen25_sae_probe.py Cell 5) makes SAE.from_pretrained
# resolve every file from disk with zero code changes to the loader itself.
import os as _os  # noqa: E402

_os.environ["HF_HOME"] = str(SAE_DIR)
from huggingface_hub import hf_hub_download  # noqa: E402

sae_files: list[Path] = []
for layer in LAYER_IDS_TO_CACHE:
    folder = SAE_FOLDER_TEMPLATE.format(layer=layer)
    # dictionary_learning-style SAE releases publish a small fixed set of
    # files per layer; ae.pt is the weights, config.json the architecture.
    for filename in ("ae.pt", "config.json"):
        remote_path = f"{folder}/{filename}"
        try:
            downloaded = hf_hub_download(repo_id=SAE_RELEASE_REPO_ID, filename=remote_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download {remote_path} from {SAE_RELEASE_REPO_ID} -- "
                "confirm the exact file layout on the HF repo page before retrying "
                "(dictionary_learning SAE repos do not all use identical filenames)."
            ) from exc
        sae_files.append(Path(downloaded))
    print(f"cached layer {layer}")

# %%
# Manifest: exact requirement/layer list, file names, sizes, SHA-256.
manifest = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "requested_packages": PACKAGE_SPECS,
    "sae_release_repo_id": SAE_RELEASE_REPO_ID,
    "sae_layers_cached": LAYER_IDS_TO_CACHE,
    "wheel_files": [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in (*wheel_files, *sdist_files)
    ],
    "sae_files": [
        {
            "name": str(p.relative_to(SAE_DIR)),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sae_files
    ],
}
manifest["total_bytes"] = sum(f["bytes"] for f in manifest["wheel_files"]) + sum(
    f["bytes"] for f in manifest["sae_files"]
)
(ASSETS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"assets ready: {ASSETS_DIR}")
print(
    f"{len(wheel_files)} wheels + {len(sdist_files)} sdists + {len(sae_files)} SAE files, "
    f"{manifest['total_bytes'] / 1024**2:.1f} MiB total"
)
print("Save this output directory as a Kaggle Dataset, then point")
print("qwen25_sae_probe.py's offline-asset search at it.")
