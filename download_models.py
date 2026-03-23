
import os
import sys
import argparse
import shutil

HF_REPO_ID = "BoredVoidAter/piepmatz-polizei"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "data", "models")

# Maps new local filename stem -> old remote filename stem on HuggingFace.
# Needed because models were originally uploaded without the ch-N and size-N
# segments that train.py now includes in every experiment name.
LEGACY_STEM_MAP = {
    "efficientnet_task-species_ch-3_size-224_attn-False_layers-3":
        "efficientnet_task-species_attn-False_layers-3",
    "efficientnet_task-type_ch-3_size-224_attn-False_layers-3":
        "efficientnet_task-type_attn-False_layers-3",
    "resnet_task-species_ch-3_size-224_attn-False_layers-3":
        "resnet_task-species_attn-False_layers-3",
    "custom_task-species_ch-3_size-224_attn-False_layers-2":
        "custom_task-species_attn-False_layers-2",
    "custom_task-species_ch-3_size-224_attn-False_layers-3":
        "custom_task-species_attn-False_layers-3",
    "custom_task-species_ch-3_size-224_attn-False_layers-5":
        "custom_task-species_attn-False_layers-5",
    "custom_task-species_ch-3_size-224_attn-True_layers-3":
        "custom_task-species_attn-True_layers-3",
    "embedding_task-species_ch-3_size-224_attn-False_layers-3":
        "embedding_task-species_attn-False_layers-3",
    "efficientnet_task-species_ch-1_size-224_attn-False_layers-3":
        "efficientnet_task-species_ch-1_attn-False_layers-3",
    "mobilenet_task-species_ch-1_size-224_attn-False_layers-3":
        "mobilenet_task-species_ch-1_attn-False_layers-3",
}

ALL_MODELS = [
    "efficientnet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "efficientnet_task-species_ch-1_size-224_attn-False_layers-3_best.pth",
    "efficientnet_task-type_ch-3_size-224_attn-False_layers-3_best.pth",
    "resnet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "custom_task-species_ch-3_size-224_attn-False_layers-2_best.pth",
    "custom_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "custom_task-species_ch-3_size-224_attn-False_layers-5_best.pth",
    "custom_task-species_ch-3_size-224_attn-True_layers-3_best.pth",
    "embedding_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "mobilenet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "mobilenet_task-species_ch-1_size-224_attn-False_layers-3_best.pth",
    "mobilenet_task-species_ch-3_size-160_attn-False_layers-3_best.pth",
    "mobilenet_task-species_ch-3_size-128_attn-False_layers-3_best.pth",
    "mobilenet_task-species_ch-3_size-96_attn-False_layers-3_best.pth",
]

SHORTCUTS = {
    "efficientnet":      "efficientnet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "efficientnet-1ch":  "efficientnet_task-species_ch-1_size-224_attn-False_layers-3_best.pth",
    "efficientnet-type": "efficientnet_task-type_ch-3_size-224_attn-False_layers-3_best.pth",
    "resnet":            "resnet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "custom-2":          "custom_task-species_ch-3_size-224_attn-False_layers-2_best.pth",
    "custom-3":          "custom_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "custom-5":          "custom_task-species_ch-3_size-224_attn-False_layers-5_best.pth",
    "custom-attn":       "custom_task-species_ch-3_size-224_attn-True_layers-3_best.pth",
    "embedding":         "embedding_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "mobilenet":         "mobilenet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",
    "mobilenet-3ch":     "mobilenet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",  # alias for clarity
    "mobilenet-1ch":     "mobilenet_task-species_ch-1_size-224_attn-False_layers-3_best.pth",
    "mobilenet-224":     "mobilenet_task-species_ch-3_size-224_attn-False_layers-3_best.pth",  # alias for clarity
    "mobilenet-160":     "mobilenet_task-species_ch-3_size-160_attn-False_layers-3_best.pth",
    "mobilenet-128":     "mobilenet_task-species_ch-3_size-128_attn-False_layers-3_best.pth",
    "mobilenet-96":      "mobilenet_task-species_ch-3_size-96_attn-False_layers-3_best.pth",
}

def get_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download
    return hf_hub_download

def _stem_and_suffix(filename):
    """Split a model filename into its experiment stem and file suffix."""
    for suffix in ("_best.pth", "_classes.json", "_arcface.pth"):
        if filename.endswith(suffix):
            return filename[:-len(suffix)], suffix
    return filename, ""

def download(hf_hub_download, local_filename, force=False):
    dest = os.path.join(MODELS_DIR, local_filename)
    if os.path.exists(dest) and not force:
        mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✓ {local_filename} ({mb:.1f} MB) — already exists, skipping")
        return True

    # Resolve the remote filename via the legacy-name mapping when needed
    stem, suffix = _stem_and_suffix(local_filename)
    remote_stem = LEGACY_STEM_MAP.get(stem, stem)
    remote_filename = remote_stem + suffix

    print(f"  ↓ {local_filename}...")
    try:
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=remote_filename,
                                          repo_type="model", local_dir=MODELS_DIR)
        # If the remote name differs from the desired local name, rename the file
        if os.path.abspath(downloaded_path) != os.path.abspath(dest):
            shutil.copy2(downloaded_path, dest)
        mb = os.path.getsize(dest) / 1024 / 1024
        print(f"    ✓ saved ({mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    ✗ failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(SHORTCUTS.keys()))
    parser.add_argument("--list",  action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\nModels at https://huggingface.co/{HF_REPO_ID}\n")
        for k, v in SHORTCUTS.items():
            exists = "✓" if os.path.exists(os.path.join(MODELS_DIR, v)) else " "
            print(f"  [{exists}] --model {k:<20} → {v}")
        return

    print(f"\n Piepmatz Polizei — Model Downloader")
    print(f"   {HF_REPO_ID}\n")
    os.makedirs(MODELS_DIR, exist_ok=True)
    hf = get_hf()

    pth_files = [SHORTCUTS[args.model]] if args.model else ALL_MODELS

    # Build the full download list: each .pth plus its companion _classes.json
    files = []
    for pth in pth_files:
        files.append(pth)
        stem, _ = _stem_and_suffix(pth)
        files.append(stem + "_classes.json")

    ok = sum(download(hf, f, args.force) for f in files)
    failed = len(files) - ok

    print(f"\n {ok} downloaded, {failed} failed.")
    if not failed:
        print("   Ready to run: python src/showcase.py --model efficientnet --live")

if __name__ == "__main__":
    main()
