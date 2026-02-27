import torch
from models import get_model
from config import *

def export_to_onnx():
    print("Lade trainiertes Modell...")
    model = get_model("efficientnet", len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "efficientnet_best.pth"), map_location="cpu"))
    model.eval()

    # Ein Dummy-Input (Batch Size 1, 3 Kanäle, 224x224)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = os.path.join(MODELS_DIR, "bird_model_quantized.onnx")

    print("Exportiere nach ONNX...")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Erfolgreich gespeichert unter: {onnx_path}")
    print("TIPP FÜR DEN PI: Installiere 'onnxruntime' statt 'torch'. Das spart extrem viel RAM!")

if __name__ == "__main__":
    export_to_onnx()
