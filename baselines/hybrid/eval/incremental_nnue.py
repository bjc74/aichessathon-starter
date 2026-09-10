import chess
from residual_nnue_inference import (fc1_weights_T,fc1_bias)
def build_accumulator(board):
    accumulator = fc1_bias.copy()