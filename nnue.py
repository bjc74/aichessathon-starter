import chess
import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 773 #12*64 + 5 - 768 piece square features, 4 castling rights and one side to move

def convert_board(board):
    '''
    Convert a chess board into a 773 element feature-array
    '''
    x = np.zeros(INPUT_SIZE, dtype = np.float32)
    #populate array x with current information
    for square, piece in board.piece_map().items():
        piece_index = piece.piece_type - 1
        if piece.color == chess.BLACK:
            piece_index += 6
        feature_index = piece_index * 64 + square
        x[feature_index] = 1.0
    #side to move:
    if board.turn == chess.WHITE:
        x[772] = 1.0
    #castling rights:
    if board.has_kingside_castling_rights(chess.WHITE):
        x[768] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[769] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[770] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[771] = 1.0
    return x
#initialising the NNUE Net
class NNUENet(nn.Module):
    def __init__(self):
        super().__init__()
        #cut down from 773 array to o/p
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        #pass x through layer then clipped relu
        x = self.fc1(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
if __name__ == "__main__":

    import time

    # Use one CPU thread to better reflect competition conditions
    torch.set_num_threads(1)

    # Create starting position
    board = chess.Board()

    # Convert board to 773-feature vector
    x = convert_board(board)

    print("Feature shape:", x.shape)
    print("Active features:", np.sum(x))

    # Create model
    model = NNUENet()
    model.eval()

    # Convert one position into a batch of size 1
    x_tensor = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0)

    print("Tensor shape:", x_tensor.shape)

    # Single inference sanity check
    with torch.inference_mode():
        output = model(x_tensor)

    print("Output shape:", output.shape)
    print("Evaluation:", output.item())

    # Warm up PyTorch
    with torch.inference_mode():
        for _ in range(1000):
            model(x_tensor)

    # Benchmark repeated inference
    NUM_EVALS = 10000

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(NUM_EVALS):
            model(x_tensor)

    elapsed = time.perf_counter() - start

    print(f"{NUM_EVALS:,} evals:", elapsed)
    print("Time per eval:", elapsed / NUM_EVALS)
    print("Evals/sec:", NUM_EVALS / elapsed)