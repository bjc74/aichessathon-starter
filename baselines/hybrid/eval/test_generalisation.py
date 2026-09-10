import chess
import numpy as np
import torch

from datasets import load_dataset
from nnue import NNUENet, convert_board


SKIP_POSITIONS = 1_002_000
TEST_POSITIONS = 5_000

CP_SCALE = 300.0
def stream_best_positions(dataset):

    current_fen = None
    best_row = None

    for row in dataset:

        if row["cp"] is None:
            continue

        fen = row["fen"]

        if current_fen is None:
            current_fen = fen
            best_row = row

        if (
            fen == current_fen
            and row["depth"] > best_row["depth"]
        ):
            best_row = row

        if fen != current_fen:

            yield best_row

            current_fen = fen
            best_row = row

    if best_row is not None:
        yield best_row
def convert_target(cp):
    return np.tanh(cp / CP_SCALE)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "best_nnue.pth"
)

model = NNUENet()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
dataset = load_dataset(
    "Lichess/chess-position-evaluations",
    split="train",
    streaming=True
)

dataset = dataset.select_columns(
    ["fen", "depth", "cp"]
)
test_boards = []
test_targets = []

for position_index, data in enumerate(
    stream_best_positions(dataset)
):

    # Ignore the entire original train/validation region
    if position_index < SKIP_POSITIONS:
        continue

    if len(test_boards) >= TEST_POSITIONS:
        break

    try:
        board = chess.Board(data["fen"])
        cp = data["cp"]

    except (KeyError, TypeError, ValueError):
        continue

    # Same side-to-move convention as training
    if board.turn == chess.BLACK:
        cp = -cp

    target = convert_target(cp)

    test_boards.append(
        convert_board(board)
    )

    test_targets.append(target)

    if len(test_boards) % 500 == 0:
        print(
            f"Collected "
            f"{len(test_boards):,}/"
            f"{TEST_POSITIONS:,}"
        )
X_test = torch.tensor(
    np.asarray(test_boards),
    dtype=torch.float32
)

y_test = np.asarray(
    test_targets,
    dtype=np.float32
)
predictions = []

BATCH_SIZE = 256

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        X_batch = X_test[
            start:start + BATCH_SIZE
        ]

        output = model(X_batch)

        predictions.append(
            output.numpy().reshape(-1)
        )

predictions = np.concatenate(predictions)
mse = np.mean(
    (predictions - y_test) ** 2
)

mae = np.mean(
    np.abs(predictions - y_test)
)

pearson = np.corrcoef(
    predictions,
    y_test
)[0, 1]

baseline_mse = np.mean(
    (y_test - y_test.mean()) ** 2
)

r2 = 1.0 - (
    np.sum((y_test - predictions) ** 2)
    /
    np.sum((y_test - y_test.mean()) ** 2)
)

print()
print("=" * 60)
print("LATER GENERALISATION SET")
print("=" * 60)

print("Positions:", len(y_test))
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Baseline MSE: {baseline_mse:.6f}")
print(f"R²: {r2:.4f}")
print(f"Pearson: {pearson:.4f}")