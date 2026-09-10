import chess
import torch
from pathlib import Path
import math
import time
import sys
from nnue import NNUENet, convert_board
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from hce import evaluate as evaluate_hce
CP_SCALE = 300.0

torch.set_num_threads(1)
CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "best_nnue.pth"
)

model = NNUENet()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
def evaluate_nnue(board):
    features = convert_board(board)
    x = torch.tensor(
    features,
    dtype=torch.float32
    ).unsqueeze(0)
    with torch.inference_mode():
        output = model(x)
    prediction = output.item()
    prediction = max(-0.999999, min(0.999999, prediction))
    cp_score = CP_SCALE * math.atanh(prediction)
    cp_score = max(-1000, min(1000, cp_score))
    return int(cp_score)
white_to_move = chess.Board(
    "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"
)

black_to_move = chess.Board(
    "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1"
)

print("White to move:", evaluate_nnue(white_to_move))
print("Black to move:", evaluate_nnue(black_to_move))