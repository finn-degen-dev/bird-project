#!/usr/bin/env bash
# =============================================================================
# upload_models.sh
# Uploads new/updated .pth files to Hugging Face and pushes code to GitHub.
#
# Usage:
#   ./upload_models.sh                         # upload all .pth files
#   ./upload_models.sh efficientnet            # upload one model by shortname
#   ./upload_models.sh efficientnet resnet     # upload multiple by shortname
#   ./upload_models.sh --pth-only              # skip the git push
#   ./upload_models.sh --git-only              # skip the HF upload
# =============================================================================

set -e

HF_REPO_ID="BoredVoidAter/piepmatz-polizei"
MODELS_DIR="data/models"

# Map shortnames → filename patterns (partial match is fine)
declare -A SHORTCUTS=(
    ["efficientnet"]="efficientnet_task-species_attn-False_layers-3_best.pth"
    ["efficientnet-type"]="efficientnet_task-type_attn-False_layers-3_best.pth"
    ["resnet"]="resnet_task-species_attn-False_layers-3_best.pth"
    ["custom-2"]="custom_task-species_attn-False_layers-2_best.pth"
    ["custom-3"]="custom_task-species_attn-False_layers-3_best.pth"
    ["custom-5"]="custom_task-species_attn-False_layers-5_best.pth"
    ["custom-attn"]="custom_task-species_attn-True_layers-3_best.pth"
    ["embedding"]="embedding_task-species_attn-False_layers-3_best.pth"
)

# ── Arg parsing ───────────────────────────────────────────────────────────────
DO_HF=true
DO_GIT=true
SPECIFIC_FILES=()

for arg in "$@"; do
    case "$arg" in
        --pth-only)  DO_GIT=false ;;
        --git-only)  DO_HF=false ;;
        --help|-h)
            echo "Usage: ./upload_models.sh [shortname...] [--pth-only] [--git-only]"
            echo ""
            echo "Shortnames:"
            for k in "${!SHORTCUTS[@]}"; do
                echo "  $k  →  ${SHORTCUTS[$k]}"
            done
            exit 0
            ;;
        -*)
            echo "Unknown flag: $arg (use --help)"
            exit 1
            ;;
        *)
            if [[ -v SHORTCUTS["$arg"] ]]; then
                SPECIFIC_FILES+=("${SHORTCUTS[$arg]}")
            else
                echo "  ✗ Unknown shortname '$arg'. Valid options:"
                for k in "${!SHORTCUTS[@]}"; do echo "    $k"; done
                exit 1
            fi
            ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Piepmatz Polizei — Model Uploader             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── HuggingFace upload ────────────────────────────────────────────────────────
if $DO_HF; then
    echo "▶ Uploading models to Hugging Face..."

    python -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub -q

    # Build the file list to pass into python
    if [ ${#SPECIFIC_FILES[@]} -gt 0 ]; then
        FILES_JSON=$(python -c "
import json, sys
files = sys.argv[1:]
print(json.dumps(files))
" "${SPECIFIC_FILES[@]}")
    else
        FILES_JSON="[]"  # empty = upload all
    fi

    python - <<EOF
import os, glob, json
from huggingface_hub import HfApi

api = HfApi()
models_dir = "${MODELS_DIR}"
specific = json.loads("""${FILES_JSON}""")

if specific:
    files = [os.path.join(models_dir, f) for f in specific]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        for m in missing:
            print(f"  ✗ File not found: {m}")
        raise SystemExit(1)
else:
    files = glob.glob(os.path.join(models_dir, "*.pth"))

if not files:
    print("  ⚠ No .pth files found — nothing to upload.")
else:
    print(f"  Uploading {len(files)} file(s) to {api.whoami()['name']}/{os.path.basename('${HF_REPO_ID}')}...")
    print()
    for path in sorted(files):
        name = os.path.basename(path)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ↑ {name} ({mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id="${HF_REPO_ID}",
            repo_type="model",
        )
        print(f"    ✓ done")
    print()
    print(f"  ✓ Upload complete → https://huggingface.co/${HF_REPO_ID}")
EOF
    echo ""
fi

# ── Git push ──────────────────────────────────────────────────────────────────
if $DO_GIT; then
    echo "▶ Pushing code changes to GitHub..."

    # Stage everything except .pth (already gitignored, but be explicit)
    git add --all -- ':!data/models/*.pth' ':!data/models/*.onnx'

    if git diff --cached --quiet; then
        echo "  (nothing new to commit — working tree clean)"
    else
        echo "  Staged changes:"
        git diff --cached --name-only | sed 's/^/    /'
        echo ""
        # Use provided message or auto-generate one from changed files
        if [ -n "$COMMIT_MSG" ]; then
            MSG="$COMMIT_MSG"
        else
            CHANGED=$(git diff --cached --name-only | head -5 | tr '\n' ' ')
            MSG="update models and code: ${CHANGED}"
        fi
        git commit -m "$MSG"
    fi

    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git push origin "${BRANCH}"
    echo "  ✓ Pushed to GitHub (${BRANCH})"
    echo ""
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║                      All done!                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Models : https://huggingface.co/${HF_REPO_ID}"
echo "  GitHub : $(git remote get-url origin)"
echo ""
