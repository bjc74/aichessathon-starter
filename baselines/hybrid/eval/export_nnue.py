import torch
from pathlib import Path
from nnue import NNUENet
ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "best_nnue.pth"
)

ONNX_PATH = (
    ROOT / "checkpoints" / "best_nnue.onnx"
)
model = NNUENet()
checkpoint = torch.load(CHECKPOINT_PATH, map_location = 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
input_tensor = torch.zeros(1,773,dtype=torch.float32)
torch.onnx.export(model,input_tensor,ONNX_PATH,input_names = ['input'], output_names = ['output'],opset_version = 17,dynamo = False)

print(f"Exported to: {ONNX_PATH}")
print(f"Exists: {ONNX_PATH.exists()}")