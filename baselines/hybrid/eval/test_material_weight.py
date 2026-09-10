import chess
import numpy as np
import torch

from pathlib import Path
from datasets import load_dataset

from nnue import NNUENet, convert_board


# ============================================================
# SETTINGS
# ============================================================

SKIP_POSITIONS = 1_002_000
TEST_POSITIONS = 5_000

CP_SCALE = 300.0
BATCH_SIZE = 256

MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "best_nnue.pth"
)


# ============================================================
# DATA HELPERS
# ============================================================

def convert_target(cp):
    return np.tanh(cp / CP_SCALE)


def stream_best_positions(dataset):
    """
    Keep only the deepest Stockfish evaluation for each FEN.
    """

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


def material_score(board):
    """
    Material score from side-to-move perspective.
    """

    score = 0

    for piece_type, value in MATERIAL_VALUES.items():

        white_count = len(
            board.pieces(
                piece_type,
                chess.WHITE
            )
        )

        black_count = len(
            board.pieces(
                piece_type,
                chess.BLACK
            )
        )

        # Initially White perspective
        score += value * (
            white_count - black_count
        )

    # Convert to side-to-move perspective
    if board.turn == chess.BLACK:
        score = -score

    return score


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = NNUENet()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# LOAD LATER HOLDOUT SET
# ============================================================

print("Loading dataset...")

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
test_materials = []


print(
    f"Skipping first "
    f"{SKIP_POSITIONS:,} positions..."
)


for position_index, data in enumerate(
    stream_best_positions(dataset)
):

    # Completely skip the region used for
    # original training + validation
    if position_index < SKIP_POSITIONS:
        continue

    if len(test_boards) >= TEST_POSITIONS:
        break

    try:

        board = chess.Board(
            data["fen"]
        )

        cp = data["cp"]

    except (
        KeyError,
        TypeError,
        ValueError
    ):
        continue

    # Convert Stockfish White POV
    # into side-to-move POV
    if board.turn == chess.BLACK:
        cp = -cp

    target = convert_target(cp)

    test_boards.append(
        convert_board(board)
    )

    test_targets.append(
        target
    )

    test_materials.append(
        material_score(board)
    )

    if len(test_boards) % 500 == 0:

        print(
            f"Collected "
            f"{len(test_boards):,}/"
            f"{TEST_POSITIONS:,}"
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

X_test = torch.tensor(
    np.asarray(test_boards),
    dtype=torch.float32
)

y_test = np.asarray(
    test_targets,
    dtype=np.float32
)

materials = np.asarray(
    test_materials,
    dtype=np.float32
)


# ============================================================
# RUN NNUE
# ============================================================

print()
print("Running NNUE...")

predictions = []


with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        X_batch = X_test[
            start:
            start + BATCH_SIZE
        ]

        output = model(
            X_batch
        )

        predictions.append(
            output
            .numpy()
            .reshape(-1)
        )


predictions = np.concatenate(
    predictions
)


# ============================================================
# NNUE OUTPUT -> CENTIPAWNS
# ============================================================

predictions_clipped = np.clip(
    predictions,
    -0.999999,
    0.999999
)

nnue_cp = (
    CP_SCALE
    * np.arctanh(
        predictions_clipped
    )
)

# Match evaluate_nnue()
nnue_cp = np.clip(
    nnue_cp,
    -1000,
    1000
)


# ============================================================
# TEST MATERIAL WEIGHTS
# ============================================================

weights_to_test = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
]


print()
print("=" * 70)
print("MATERIAL WEIGHT SWEEP")
print("=" * 70)

print(
    f"{'Weight':>8}"
    f"{'MSE':>12}"
    f"{'MAE':>12}"
    f"{'R2':>12}"
    f"{'Pearson':>12}"
)

print("-" * 56)


best_weight = None
best_mse = float("inf")


for weight in weights_to_test:

    # This exactly represents:
    #
    # hybrid =
    # NNUE + weight * material
    #
    # in centipawns.
    hybrid_cp = (
        nnue_cp
        + weight * materials
    )

    # Convert back into the target space
    # so we can compare against y_test.
    hybrid_prediction = np.tanh(
        hybrid_cp / CP_SCALE
    )

    mse = float(
        np.mean(
            (
                hybrid_prediction
                - y_test
            ) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                hybrid_prediction
                - y_test
            )
        )
    )

    ss_res = float(
        np.sum(
            (
                y_test
                - hybrid_prediction
            ) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_test
                - y_test.mean()
            ) ** 2
        )
    )

    r2 = (
        1.0
        - ss_res / ss_tot
    )

    pearson = float(
        np.corrcoef(
            hybrid_prediction,
            y_test
        )[0, 1]
    )

    print(
        f"{weight:8.1f}"
        f"{mse:12.6f}"
        f"{mae:12.6f}"
        f"{r2:12.4f}"
        f"{pearson:12.4f}"
    )

    if mse < best_mse:

        best_mse = mse
        best_weight = weight


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)

print(
    "Best material weight:",
    best_weight
)

print(
    "Best MSE:",
    f"{best_mse:.6f}"
)

print("=" * 70)