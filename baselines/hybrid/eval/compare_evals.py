import os
import sys
from pathlib import Path

import chess
import numpy as np
import torch
from datasets import load_dataset

# hce.py and hce_v3.py are in the repo root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nnue import NNUENet, convert_board
from hce import evaluate as evaluate_hce_v2
from hce_v3 import evaluate as evaluate_hce_v3


VALIDATION_SIZE = 2_000
MAX_SAMPLES = 1_002_000
CP_SCALE = 300.0

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "best_nnue.pth"
)


def stream_best_positions(dataset):

    current_fen = None
    best_row = None

    for row in dataset:

        # Skip mate-labelled positions
        if row["cp"] is None:
            continue

        fen = row["fen"]

        if current_fen is None:
            current_fen = fen
            best_row = row

        # For repeated rows of the same FEN,
        # keep the deepest Stockfish evaluation.
        if fen == current_fen and row["depth"] > best_row["depth"]:
            best_row = row

        # We have reached a new position:
        # yield the best row for the previous one.
        if fen != current_fen:
            yield best_row
            current_fen = fen
            best_row = row

    if best_row is not None:
        yield best_row


def load_position_dataset():

    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True
    )

    return dataset.select_columns(
        ["fen", "depth", "cp"]
    )


def metrics(name, predictions, targets):

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    mse = np.mean(
        (predictions - targets) ** 2
    )

    mae = np.mean(
        np.abs(predictions - targets)
    )

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - targets.mean()) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    pearson = np.corrcoef(
        predictions,
        targets
    )[0, 1]

    print()
    print(name)
    print(f"MSE     = {mse:.6f}")
    print(f"MAE     = {mae:.6f}")
    print(f"R2      = {r2:.4f}")
    print(f"Pearson = {pearson:.4f}")


# ---------------------------------------------------------
# Recreate exact same validation positions as NNUE training
# ---------------------------------------------------------

rng = np.random.default_rng(42)

validation_indices = set(
    rng.choice(
        MAX_SAMPLES,
        size=VALIDATION_SIZE,
        replace=False
    ).tolist()
)


targets = []

hce_v2_predictions = []
hce_v3_predictions = []

nnue_features = []

validation_fens = []
stockfish_cps = []


dataset = load_position_dataset()

print("Loading validation positions...")


for position_index, data in enumerate(
    stream_best_positions(dataset)
):

    if position_index >= MAX_SAMPLES:
        break

    if position_index not in validation_indices:
        continue

    board = chess.Board(data["fen"])
    cp = data["cp"]

    # Training target was side-to-move perspective
    if board.turn == chess.BLACK:
        cp = -cp

    validation_fens.append(data["fen"])
    stockfish_cps.append(cp)

    target = np.tanh(cp / CP_SCALE)
    targets.append(target)

    # -----------------------------------------------------
    # HCE v2 + v3
    # Both HCEs return White-perspective scores
    # -----------------------------------------------------

    hce_v2_cp = evaluate_hce_v2(board)
    hce_v3_cp = evaluate_hce_v3(board)

    # Convert HCEs to side-to-move perspective
    if board.turn == chess.BLACK:
        hce_v2_cp = -hce_v2_cp
        hce_v3_cp = -hce_v3_cp

    hce_v2_predictions.append(
        np.tanh(hce_v2_cp / CP_SCALE)
    )

    hce_v3_predictions.append(
        np.tanh(hce_v3_cp / CP_SCALE)
    )

    # NNUE inputs
    nnue_features.append(
        convert_board(board)
    )


print(f"Loaded {len(targets):,} validation positions")

assert len(targets) == VALIDATION_SIZE
assert len(hce_v2_predictions) == VALIDATION_SIZE
assert len(hce_v3_predictions) == VALIDATION_SIZE
assert len(nnue_features) == VALIDATION_SIZE


# ---------------------------------------------------------
# Load NNUE
# ---------------------------------------------------------

torch.set_num_threads(1)

model = NNUENet()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


X = torch.tensor(
    np.asarray(nnue_features),
    dtype=torch.float32
)

with torch.inference_mode():

    nnue_predictions = (
        model(X)
        .squeeze(1)
        .numpy()
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

targets = np.asarray(targets)

metrics(
    "HCE v2",
    hce_v2_predictions,
    targets
)

metrics(
    "HCE v3",
    hce_v3_predictions,
    targets
)

metrics(
    "NNUE",
    nnue_predictions,
    targets
)


# ---------------------------------------------------------
# Inspect genuinely decisive held-out positions
# ---------------------------------------------------------

print()
print("=" * 70)
print("REAL VALIDATION POSITIONS WITH |STOCKFISH| >= 600 CP")
print("=" * 70)

shown = 0

for i in range(len(targets)):

    if abs(stockfish_cps[i]) < 600:
        continue

    board = chess.Board(validation_fens[i])

    # NNUE output is tanh(cp / 300)
    prediction = float(nnue_predictions[i])

    prediction = max(
        -0.999999,
        min(0.999999, prediction)
    )

    nnue_cp = CP_SCALE * np.arctanh(prediction)

    # HCE v2 in side-to-move perspective
    hce_cp = evaluate_hce_v2(board)

    if board.turn == chess.BLACK:
        hce_cp = -hce_cp

    print()
    print(f"Position {shown + 1}")
    print("FEN:", validation_fens[i])
    print(f"Stockfish: {stockfish_cps[i]:+.0f} cp")
    print(f"HCE v2:    {hce_cp:+d} cp")
    print(f"NNUE:      {nnue_cp:+.0f} cp")

    shown += 1

    if shown == 20:
        break