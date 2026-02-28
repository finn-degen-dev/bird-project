
"""
Download pre-trained model weights from Hugging Face.

Usage:
    python download_models.py               # all models
    python download_models.py --model efficientnet
    python download_models.py --list
    python download_models.py --force       # re-download existing
"""
import os, sys, argparse

HF_REPO_ID = "BoredVoidAter/piepmatz-polizei"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "data", "models")

ALL_MODELS = [
    "efficientnet_task-species_attn-False_layers-3_best.pth",
    "efficientnet_task-type_attn-False_layers-3_best.pth",
    "resnet_task-species_attn-False_layers-3_best.pth",
    "custom_task-species_attn-False_layers-2_best.pth",
    "custom_task-species_attn-False_layers-3_best.pth",
    "custom_task-species_attn-False_layers-5_best.pth",
    "custom_task-species_attn-True_layers-3_best.pth",
    "embedding_task-species_attn-False_layers-3_best.pth",
]

SHORTCUTS = {
    "efficientnet":      "efficientnet_task-species_attn-False_layers-3_best.pth",
    "efficientnet-type": "efficientnet_task-type_attn-False_layers-3_best.pth",
    "resnet":            "resnet_task-species_attn-False_layers-3_best.pth",
    "custom-2":          "custom_task-species_attn-False_layers-2_best.pth",
    "custom-3":          "custom_task-species_attn-False_layers-3_best.pth",
    "custom-5":          "custom_task-species_attn-False_layers-5_best.pth",
    "custom-attn":       "custom_task-species_attn-True_layers-3_best.pth",
    "embedding":         "embedding_task-species_attn-False_layers-3_best.pth",
}

def get_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download
    return hf_hub_download

def download(hf_hub_download, filename, force=False):
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest) and not force:
        mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✓ {filename} ({mb:.1f} MB) — already exists, skipping")
        return True
    print(f"  ↓ {filename}...")
    try:
        hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                        repo_type="model", local_dir=MODELS_DIR)
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

    files = [SHORTCUTS[args.model]] if args.model else ALL_MODELS
    ok = sum(download(hf, f, args.force) for f in files)
    failed = len(files) - ok

    print(f"\n {ok} downloaded, {failed} failed.")
    if not failed:
        print("   Ready to run: python src/showcase.py --model efficientnet --live")

if __name__ == "__main__":
    main()
