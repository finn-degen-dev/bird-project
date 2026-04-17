import torch
import json
import os
import sys

# Add src to path so we can import your model builder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models import get_model

# Names based on your 'ls' output
exp_name = "efficientnet_task-type_attn-False_layers-3"
json_path = f"data/models/{exp_name}_classes.json"
pth_path = f"data/models/{exp_name}_best.pth"
onnx_path = "data/models/efficientnet_type_fp32.onnx"

print(f"Loading classes from {json_path}...")
with open(json_path, "r") as f:
    classes = json.load(f)["classes"]

print("Loading PyTorch model...")
model = get_model("efficientnet", len(classes), use_attention=False, num_layers=3, in_channels=3).to("cpu")
model.load_state_dict(torch.load(pth_path, map_location="cpu"))
model.eval()

print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, onnx_path, opset_version=17, dynamo=False)

print(f"Done! Saved to {onnx_path}")
