import os
import torch
import json
import onnx
import gc
from torch.utils.data import DataLoader
from models import get_model
from config import *
from data import SmartBirdDataset
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, CalibrationMethod, shape_inference

class BirdDataReader(CalibrationDataReader):
    def __init__(self, limit=150):

        dataset = SmartBirdDataset(mode='test', task='species')
        self.dataloader = iter(DataLoader(dataset, batch_size=1, shuffle=True))
        self.limit = limit
        self.count = 0

    def get_next(self):
        if self.count >= self.limit:
            return None
        try:
            img, _ = next(self.dataloader)
            self.count += 1
            return {"input": img.numpy()}
        except StopIteration:
            return None

def main():
    model_name = "efficientnet"
    exp_name = f"{model_name}_task-species_attn-False_layers-3"

    onnx_fp32_path = os.path.join(MODELS_DIR, f"{model_name}_fp32.onnx")
    onnx_preprocessed_path = os.path.join(MODELS_DIR, f"{model_name}_preprocessed.onnx")
    onnx_int8_path = os.path.join(MODELS_DIR, f"{model_name}_quantized_int8_entropy.onnx")

    if not os.path.exists(onnx_fp32_path):
        print("Lade PyTorch Modell...")
        with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
            num_classes = len(json.load(f)["classes"])
        model = get_model(model_name, num_classes).to("cpu")
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location="cpu"))
        model.eval()
        torch.onnx.export(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), onnx_fp32_path, opset_version=17, dynamo=False)
        del model
        gc.collect()

    print("Führe ONNX Pre-Processing durch...")
    shape_inference.quant_pre_process(onnx_fp32_path, onnx_preprocessed_path, skip_symbolic_shape=False)

    print(f"Starte Quantisierung (Percentile Methode, 150 Samples)...")
    data_reader = BirdDataReader(limit=150)

    quantize_static(
        model_input=onnx_preprocessed_path,
        model_output=onnx_int8_path,
        calibration_data_reader=data_reader,
        quant_format=0,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            'ActivationSymmetry': False,
            'WeightSymmetry': True,
            'Percentile': 99.99
        },
        per_channel=False
    )

    if os.path.exists(onnx_preprocessed_path):
        os.remove(onnx_preprocessed_path)

    print("\nFertig! Bitte benchmark_quantization.py erneut ausführen.")

if __name__ == "__main__":
    main()
