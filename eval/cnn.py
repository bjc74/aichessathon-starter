"""
Chess position evaluator.

Input:
    13 x 8 x 8 tensor

Planes:
    0  = white pawns
    1  = white knights
    2  = white bishops
    3  = white rooks
    4  = white queens
    5  = white king

    6  = black pawns
    7  = black knights
    8  = black bishops
    9  = black rooks
    10 = black queens
    11 = black king

    12 = side to move
          all 1s if White to move
          all 0s if Black to move

Output:
    A single scalar evaluation from the perspective
    of the side to move.
"""

import torch
import torch.nn as nn
import numpy as np
import chess

def convert_board(board):
    """
    Convert a chess.Board into an 18 x 8 x 8 float32 tensor.

    Planes:
        0  = white pawns
        1  = white knights
        2  = white bishops
        3  = white rooks
        4  = white queens
        5  = white king

        6  = black pawns
        7  = black knights
        8  = black bishops
        9  = black rooks
        10 = black queens
        11 = black king

        12 = side to move
             all 1s if White to move
             all 0s if Black to move

        13 = white can castle kingside
        14 = white can castle queenside
        15 = black can castle kingside
        16 = black can castle queenside

        17 = en-passant target square
             1 at the target square, 0 everywhere else
    """

    x = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes

    piece_planes = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,

        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }

    for (colour, piece_type), plane in piece_planes.items():

        for square in board.pieces(piece_type, colour):

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            x[plane, row, col] = 1.0

    # Side to move

    if board.turn == chess.WHITE:
        x[12, :, :] = 1.0

    # Castling rights

    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1.0

    # En-passant target square

    if board.ep_square is not None:

        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)

        x[17, row, col] = 1.0

    return x

class ResidualBlock(nn.Module):
    """
    Standard 3x3 residual block.

    Input and output have the same shape:

        [batch, channels, 8, 8]

    The skip connection makes it easier to train a deeper
    CNN without making optimisation unnecessarily difficult.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Residual / skip connection
        x = x + residual

        x = self.relu(x)

        return x

class EvalNet(nn.Module):

    def __init__(self):
        super().__init__()

        channels = 64

        # Initial feature extraction
  
        self.input_layer = nn.Sequential(

            nn.Conv2d(
                in_channels=18,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),
        )

        # Residual tower

        self.residual_blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

        self.fc1 = nn.Linear(channels * 8 * 8, 256)

        self.fc2 = nn.Linear(256, 64)

        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Input:
        # [batch, 13, 8, 8]

        x = self.input_layer(x)

        # [batch, 64, 8, 8]

        x = self.residual_blocks(x)

        # [batch, 64, 8, 8]

        # Flatten while preserving all spatial information.
        x = x.view(x.size(0), -1)

        # [batch, 4096]

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        # [batch, 1]

        return x