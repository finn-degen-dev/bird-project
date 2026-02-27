import os
import torch
import argparse
from models import get_model
from config import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet"], default="efficientnet")
    args = parser.parse_args()

    exp_name = f"{args.model}_task-species_attn-False_layers-3"

    import json
    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        num_classes = len(json.load(f)['classes'])

    print(f"Lade {exp_name}...")
    model = get_model(args.model, num_classes).to("cpu")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = os.path.join(MODELS_DIR, f"{args.model}_quantized.onnx")

    print(f"Exportiere nach ONNX: {onnx_path}...")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(" ONNX Export erfolgreich. Perfekt für den Raspberry Pi!")

if __name__ == "__main__":
    main()
