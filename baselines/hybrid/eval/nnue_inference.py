import chess
import math
import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
from nnue import NNUENet
from nnue import convert_board
from numba import njit
ROOT = Path(__file__).resolve().parents[1]
CP_SCALE = 300
CHECKPOINT_PATH = (Path(__file__).resolve().parents[1]/ "checkpoints"/ "best_nnue.pth")
ONNX_PATH = (ROOT/ "checkpoints"/ "best_nnue.onnx")
model = NNUENet()
checkpoint = torch.load(CHECKPOINT_PATH,map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
fc1_weights = (model.fc1.weight.detach().numpy())
fc1_weights_T = np.ascontiguousarray(fc1_weights.T)
fc1_bias = (model.fc1.bias.detach().numpy())
fc2_weights = model.fc2.weight.detach().numpy()
fc2_bias = model.fc2.bias.detach().numpy()
fc3_weights = model.fc3.weight.detach().numpy()
fc3_bias = model.fc3.bias.detach().numpy()
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(ONNX_PATH),sess_options=options,providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
FEATURE_BUFFER = np.zeros((1, 773),dtype=np.float32)
def get_nnue_board_data(board):
    bitboards = np.empty(12, dtype=np.uint64)
    i = 0
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            bitboards[i] = board.pieces_mask(piece_type,colour)
            i += 1
    state = np.empty(5, dtype=np.uint8)
    state[0] = board.has_kingside_castling_rights(chess.WHITE)
    state[1] = board.has_queenside_castling_rights(chess.WHITE)
    state[2] = board.has_kingside_castling_rights(chess.BLACK)
    state[3] = board.has_queenside_castling_rights(chess.BLACK)
    state[4] = board.turn == chess.WHITE
    return bitboards, state
@njit(cache=True)
def nnue_forward_numba(bitboards, state, fc1_weights, fc1_bias, fc2_weights, fc2_bias, fc3_weights, fc3_bias):
    hidden = fc1_bias.copy()
    for piece_index in range(12):
        bb = bitboards[piece_index]
        while bb:
            lsb = bb & -bb
            square = 0
            temp = lsb
            while temp > 1:
                temp >>= 1
                square += 1
            feature_index = piece_index * 64 + square
            for j in range(256):
                hidden[j] += fc1_weights[feature_index, j]
            bb ^= lsb
    for state_index in range(5):

        if state[state_index]:

            feature_index = 768 + state_index

            for j in range(256):
                hidden[j] += fc1_weights[feature_index, j]
    # Clamp activation from the trained network
    for j in range(256):
        if hidden[j] < 0.0:
            hidden[j] = 0.0
        elif hidden[j] > 1.0:
            hidden[j] = 1.0
    hidden2 = fc2_bias.copy()
    for j in range(32):
        for i in range(256):
            hidden2[j] += fc2_weights[j, i] * hidden[i]
    for j in range(32):
        if hidden2[j] < 0.0:
            hidden2[j] = 0.0
    output = fc3_bias[0]
    for i in range(32):
        output += fc3_weights[0, i] * hidden2[i]
    return output
def convert_board_fast(board):
    x = FEATURE_BUFFER
    x.fill(0.0)
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type in range(chess.PAWN,chess.KING + 1):
            bb = board.pieces_mask(
                piece_type,
                colour
            )
            piece_index = piece_type - 1
            if colour == chess.BLACK:
                piece_index += 6
            while bb:
                lsb = bb & -bb
                square = lsb.bit_length() - 1
                feature_index = (
                    piece_index * 64
                    + square
                )
                x[0, feature_index] = 1.0
                bb ^= lsb
    if board.turn == chess.WHITE:
            x[0,772] = 1.0
    #castling rights:
    if board.has_kingside_castling_rights(chess.WHITE):
        x[0,768] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[0,769] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[0,770] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[0,771] = 1.0
    return x
        
def evaluate_nnue(board):

    bitboards, state = get_nnue_board_data(board)

    prediction = nnue_forward_numba(
        bitboards,
        state,
        fc1_weights_T,
        fc1_bias,
        fc2_weights,
        fc2_bias,
        fc3_weights,
        fc3_bias
    )
    prediction = max(-0.999999, min(0.999999, prediction))
    cp_score = CP_SCALE * math.atanh(prediction)

    cp_score = max(
    -1000,
    min(1000, cp_score)
    )

    return int(cp_score)
