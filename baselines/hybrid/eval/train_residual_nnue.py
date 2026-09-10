import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import chess
import numpy as np
import os
import time
from datasets import load_dataset#import dataset through hugging face as i don't have space for 1 billion chess positions on my computer
from nnue import convert_board, NNUENet

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "checkpoints"
)

CHECKPOINT_DIR = os.path.abspath(CHECKPOINT_DIR)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best_residual_nnue.pth"
)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

def material_score(board):
    mover = board.turn

    material = sum(
        value * (
            len(board.pieces(piece, mover))
            - len(board.pieces(piece, not mover))
        )
        for piece, value in PIECE_VALUES.items()
    )

    return material

TRAINING_SAMPLES = 1_000_000
# Fixed validation set size
VALIDATION_SIZE = 2_000
# Total number of usable positions to process
MAX_SAMPLES = TRAINING_SAMPLES + VALIDATION_SIZE

# Number of samples kept in memory at once
BUFFER_SIZE = 100_000


BATCH_SIZE = 256

EPOCHS_PER_BUFFER = 1

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 1.0

# Stockfish centipawn scaling
# Smaller values make the network care more about distinguishing
# strong evaluations such as +500 from +1000
CP_SCALE = 300.0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# Model

model = NNUENet()
model.to(device)
model.train()

optimiser = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

loss_function = nn.SmoothL1Loss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode="min",
    patience=2,
    factor=0.5
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def convert_target(cp, board):
    # Convert Stockfish centipawn evaluation to a bounded target. The target is always approximately in [-1, 1]
    return np.tanh((cp - material_score(board)) / CP_SCALE)
#make sure the same FEN isn't used multiple times, only once at it's best depth
def stream_best_positions(dataset):
    current_fen = None
    best_row = None
    for row in dataset:
        if row['cp'] is None:
            continue
        fen = row["fen"]
        if current_fen is None:
            current_fen = fen
            best_row = row
        if fen == current_fen and row['depth'] > best_row['depth']:
            best_row = row
        if fen!=current_fen:
            yield best_row
            current_fen = fen
            best_row = row
    if best_row is not None:
        yield best_row
total_samples = 0
total_batches = 0

running_loss = 0.0
running_samples = 0

best_val_mse = float("inf")

start_time = time.time()

validation_boards = []
validation_targets = []


# Training

def load_position_dataset():

    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True
    )

    return dataset.select_columns(
        ["fen", "depth", "cp"]
    )

rng = np.random.default_rng(42)  # fixed seed

validation_indices = set(
    rng.choice(
        MAX_SAMPLES,
        size=VALIDATION_SIZE,
        replace=False
    )
)

print("Building validation set...")

dataset = load_position_dataset()

validation_boards = []
validation_targets = []

for position_index, data in enumerate(
    stream_best_positions(dataset)
):

    if position_index >= MAX_SAMPLES:
        break

    if position_index not in validation_indices:
        continue

    try:
        board = chess.Board(data["fen"])
        cp = data["cp"]

    except (KeyError, TypeError, ValueError):
        continue

    if board.turn == chess.BLACK:
        cp = -cp

    score = convert_target(cp, board)

    validation_boards.append(
        convert_board(board)
    )

    validation_targets.append(score)

print(
    f"Validation set complete: "
    f"{len(validation_boards):,} positions"
)

if len(validation_boards) != VALIDATION_SIZE:
    raise RuntimeError(
        f"Expected {VALIDATION_SIZE} validation positions, "
        f"but got {len(validation_boards)}"
    )

print("Starting training...")

# Start a fresh stream from the beginning
dataset = load_position_dataset()

buffer_boards = []
buffer_targets = []

total_samples = 0

for position_index, data in enumerate(
    stream_best_positions(dataset)
):

    # Only consider the same 1,002,000 positions as Pass 1
    if position_index >= MAX_SAMPLES:
        break

    # Track how far through the source data we are
    total_samples = position_index + 1

    # Validation positions must never be used for training
    if position_index in validation_indices:
        continue

    try:
        board = chess.Board(data["fen"])
        cp = data["cp"]

    except (KeyError, TypeError, ValueError):
        continue

    # Convert Stockfish score to side-to-move perspective
    if board.turn == chess.BLACK:
        cp = -cp

    score = convert_target(cp, board)

    encoded_board = convert_board(board)

    buffer_boards.append(encoded_board)
    buffer_targets.append(score)

    # Train when buffer is full
    if len(buffer_boards) >= BUFFER_SIZE:


        # Convert buffer to tensors
        X_buf = torch.tensor(
            np.asarray(buffer_boards),
            dtype=torch.float32
        )

        y_buf = torch.tensor(
            buffer_targets,
            dtype=torch.float32
        ).unsqueeze(1)

        print()
        print("=" * 70)
        print(
            f"Training buffer "
            f"{total_samples:,}/{MAX_SAMPLES:,}"
        )

        print(f"buffer samples = {len(X_buf):,}")

        print(f"target mean = {y_buf.mean().item():.4f}")

        print(f"target std  = {y_buf.std(unbiased=False).item():.4f}")

        # Shuffle buffer
        perm = torch.randperm(len(X_buf))

        X_buf = X_buf[perm]
        y_buf = y_buf[perm]

        # DataLoader
        train_ds = TensorDataset(
            X_buf,
            y_buf
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

        # Train
        for epoch in range(EPOCHS_PER_BUFFER):

            model.train()

            epoch_loss = 0.0
            epoch_batches = 0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(
                    device,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    device,
                    non_blocking=True
                )

                optimiser.zero_grad(
                    set_to_none=True
                )

                prediction = model(X_batch)

                loss = loss_function(
                    prediction,
                    y_batch
                )

                loss.backward()

                total_norm = 0.0

                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += (param_norm.item() ** 2)

                total_norm = total_norm ** 0.5

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimiser.step()

                running_loss += loss.item()

                running_samples += len(X_batch)

                epoch_loss += loss.item()
                epoch_batches += 1

                total_batches += 1

                if total_batches % 100 == 0:

                    average_loss = (
                        running_loss / 100.0
                    )

                    print(
                        f"step={total_batches:,} "
                        f"epoch={epoch + 1}/{EPOCHS_PER_BUFFER} "
                        f"loss={average_loss:.4f} "
                        f"grad_norm={total_norm:.4f}"
                    )

                    running_loss = 0.0

            average_epoch_loss = (
                epoch_loss /
                max(epoch_batches, 1)
            )

            print(
                f"epoch {epoch + 1}/{EPOCHS_PER_BUFFER} "
                f"loss={average_epoch_loss:.4f}"
            )

        # Validation

        model.eval()

        val_preds = []

        val_targets = []

        # Convert validation set to tensors once.
        X_val = torch.tensor(
            np.asarray(validation_boards),
            dtype=torch.float32
        )

        y_val = torch.tensor(
            validation_targets,
            dtype=torch.float32
        ).unsqueeze(1)

        val_ds = TensorDataset(
            X_val,
            y_val
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

        with torch.no_grad():

            for Xv, yv in val_loader:

                Xv = Xv.to(
                    device,
                    non_blocking=True
                )

                preds = model(Xv)

                val_preds.append(
                    preds.cpu().numpy().reshape(-1)
                )

                val_targets.append(
                    yv.numpy().reshape(-1)
                )

        val_preds = np.concatenate(val_preds)

        val_targets_np = np.concatenate(val_targets)

        # Metrics

        val_mse = float(np.mean((val_preds - val_targets_np) ** 2))

        val_mae = float(np.mean(np.abs(val_preds - val_targets_np)))

        # Baseline = always predict validation mean.
        baseline_prediction = (val_targets_np.mean())

        baseline_mse = float(np.mean((val_targets_np - baseline_prediction) ** 2))

        # R²
        ss_res = float(np.sum((val_targets_np - val_preds) ** 2))

        ss_tot = float(np.sum((val_targets_np - val_targets_np.mean()) ** 2))

        r2 = (1.0 - ss_res / (ss_tot + 1e-12))

        # Pearson correlation
        if (val_preds.size > 1 and np.std(val_preds) > 0 and np.std(val_targets_np) > 0):

            pearson = float(np.corrcoef(val_preds, val_targets_np)[0, 1])

        else:
            pearson = float("nan")

        # Learning rate
        current_lr = optimiser.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print()
        print(f"VALIDATION")

        print(f"mse = {val_mse:.6f}")

        print(f"mae = {val_mae:.6f}")

        print(f"baseline_mse = {baseline_mse:.6f}")

        print(f"r2 = {r2:.4f}")

        print(f"pearson = {pearson:.4f}")

        print(f"learning_rate = {current_lr:.6g}")

        print(f"elapsed = {elapsed / 60:.1f} min")

        # Scheduler
        scheduler.step(val_mse)

        # Save best checkpoint
        if np.isfinite(val_mse):

            if val_mse < best_val_mse:

                best_val_mse = val_mse

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimiser.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mse": val_mse,
                        "val_mae": val_mae,
                        "r2": r2,
                        "pearson": pearson,
                        "y_mean_train": 0.0,
                        "y_std_train": 1.0,
                        "cp_scale": CP_SCALE,
                        "samples_seen": total_samples,
                        "batch_count": total_batches,
                    },
                    BEST_CHECKPOINT
                )

                print()
                print(f"NEW BEST MODEL")

                print(
                    f"validation MSE = "
                    f"{val_mse:.6f}"
                )

                print(
                    f"saved to: "
                    f"{BEST_CHECKPOINT}"
                )

        # Clear buffer
        del X_buf
        del y_buf

        buffer_boards.clear()
        buffer_targets.clear()

        if total_samples >= MAX_SAMPLES:
            break

# Final information
elapsed = time.time() - start_time

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"samples processed = {total_samples:,}")

print(f"batches = {total_batches:,}")

print(f"best validation MSE = {best_val_mse:.6f}")

print(f"time = {elapsed / 60:.1f} minutes")

print(f"checkpoint = {BEST_CHECKPOINT}")