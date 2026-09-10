import chess
from nnue_inference import (fc1_weights_T,fc1_bias, fc2_bias, fc2_weights, fc3_bias, fc3_weights)
from numba import njit
import math
CP_SCALE = 300
MATERIAL_VALUES = (100, 320, 330, 500, 900, 0)
def build_accumulator(board):
    accumulator = fc1_bias.copy()
    material_white = 0
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type in range(chess.PAWN, chess.KING +1):
            bb = board.pieces_mask(piece_type, colour)
            piece_value = MATERIAL_VALUES[piece_type - 1]
            piece_count = bb.bit_count()

            if colour == chess.WHITE:
                material_white += piece_value * piece_count
            else:
                material_white -= piece_value * piece_count
            piece_index = piece_type - 1
            if colour == chess.BLACK:
                piece_index += 6
            while bb:
                lsb = bb & -bb
                square = lsb.bit_length() - 1
                feature_index = piece_index * 64 + square
                accumulator += fc1_weights_T[feature_index]
                bb ^= lsb
    if board.has_kingside_castling_rights(chess.WHITE):
        accumulator+=fc1_weights_T[768]
    if board.has_queenside_castling_rights(chess.WHITE):
        accumulator+=fc1_weights_T[769]
    if board.has_kingside_castling_rights(chess.BLACK):
        accumulator+=fc1_weights_T[770]
    if board.has_queenside_castling_rights(chess.BLACK):
        accumulator+=fc1_weights_T[771]
    if board.turn == chess.WHITE:
        accumulator+=fc1_weights_T[772]
    return accumulator, material_white
@njit(cache = True)
def evaluate_from_accumulator(accumulator, material_white, white_to_move, fc2_weights, fc2_bias, fc3_weights, fc3_bias):
    hidden = accumulator.copy()
    for i in range(hidden.shape[0]):
        if hidden[i] < 0:
            hidden[i] = 0
        elif hidden[i] > 1:
            hidden[i] = 1
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
    if output > 0.999999:
        output = 0.999999
    elif output < -0.999999:
        output = -0.999999
    nnue_cp = CP_SCALE * math.atanh(output)

    if nnue_cp > 1000:
        nnue_cp = 1000
    elif nnue_cp < -1000:
        nnue_cp = -1000

    if white_to_move:
        material = material_white
    else:
        material = -material_white

    score = nnue_cp + 0.8 * material

    return int((score))
def update_accumulator(board, move, accumulator, material_white):

    # Null move only changes side to move
    if move == chess.Move.null():

        if board.turn == chess.WHITE:
            accumulator -= fc1_weights_T[772]
        else:
            accumulator += fc1_weights_T[772]

        return material_white

    white_ks = board.has_kingside_castling_rights(chess.WHITE)
    white_qs = board.has_queenside_castling_rights(chess.WHITE)
    black_ks = board.has_kingside_castling_rights(chess.BLACK)
    black_qs = board.has_queenside_castling_rights(chess.BLACK)

    piece = board.piece_at(move.from_square)

    piece_type = piece.piece_type
    colour = piece.color

    piece_index = piece_type - 1

    if colour == chess.BLACK:
        piece_index += 6

    old_feature = piece_index * 64 + move.from_square

    accumulator -= fc1_weights_T[old_feature]

    # Normal move
    if move.promotion is None:

        new_feature = piece_index * 64 + move.to_square

        accumulator += fc1_weights_T[new_feature]

    # Promotion
    else:

        promoted_index = move.promotion - 1

        promotion_gain = (
            MATERIAL_VALUES[move.promotion - 1]
            - MATERIAL_VALUES[chess.PAWN - 1]
        )

        if colour == chess.BLACK:
            promoted_index += 6
            material_white -= promotion_gain
        else:
            material_white += promotion_gain

        new_feature = promoted_index * 64 + move.to_square

        accumulator += fc1_weights_T[new_feature]

    # Capture
    if board.is_capture(move):

        if board.is_en_passant(move):

            if colour == chess.WHITE:
                captured_square = move.to_square - 8
            else:
                captured_square = move.to_square + 8

        else:
            captured_square = move.to_square

        captured_piece = board.piece_at(captured_square)

        captured_index = captured_piece.piece_type - 1
        captured_value = MATERIAL_VALUES[captured_piece.piece_type - 1]

        if captured_piece.color == chess.BLACK:
            captured_index += 6
            material_white += captured_value
        else:
            material_white -= captured_value

        captured_feature = captured_index * 64 + captured_square

        accumulator -= fc1_weights_T[captured_feature]

        # Capturing a rook can remove castling rights
        if captured_square == chess.H1 and white_ks:
            accumulator -= fc1_weights_T[768]

        elif captured_square == chess.A1 and white_qs:
            accumulator -= fc1_weights_T[769]

        elif captured_square == chess.H8 and black_ks:
            accumulator -= fc1_weights_T[770]

        elif captured_square == chess.A8 and black_qs:
            accumulator -= fc1_weights_T[771]

    # Castling also moves the rook
    if board.is_castling(move):

        rook_index = chess.ROOK - 1

        if colour == chess.BLACK:
            rook_index += 6

        if colour == chess.WHITE:

            if board.is_kingside_castling(move):
                rook_from = chess.H1
                rook_to = chess.F1
            else:
                rook_from = chess.A1
                rook_to = chess.D1

        else:

            if board.is_kingside_castling(move):
                rook_from = chess.H8
                rook_to = chess.F8
            else:
                rook_from = chess.A8
                rook_to = chess.D8

        old_rook_feature = rook_index * 64 + rook_from
        new_rook_feature = rook_index * 64 + rook_to

        accumulator -= fc1_weights_T[old_rook_feature]
        accumulator += fc1_weights_T[new_rook_feature]

    # King move removes both castling rights
    if piece_type == chess.KING:

        if colour == chess.WHITE:

            if white_ks:
                accumulator -= fc1_weights_T[768]

            if white_qs:
                accumulator -= fc1_weights_T[769]

        else:

            if black_ks:
                accumulator -= fc1_weights_T[770]

            if black_qs:
                accumulator -= fc1_weights_T[771]

    # Rook move can remove one castling right
    elif piece_type == chess.ROOK:

        if move.from_square == chess.H1 and white_ks:
            accumulator -= fc1_weights_T[768]

        elif move.from_square == chess.A1 and white_qs:
            accumulator -= fc1_weights_T[769]

        elif move.from_square == chess.H8 and black_ks:
            accumulator -= fc1_weights_T[770]

        elif move.from_square == chess.A8 and black_qs:
            accumulator -= fc1_weights_T[771]

    # Side to move feature
    if board.turn == chess.WHITE:
        accumulator -= fc1_weights_T[772]
    else:
        accumulator += fc1_weights_T[772]

    return material_white
def evaluate_nnue_rebuild(board):
    accumulator, material_white = build_accumulator(board)

    return evaluate_from_accumulator(
        accumulator,
        material_white,
        board.turn == chess.WHITE,
        fc2_weights,
        fc2_bias,
        fc3_weights,
        fc3_bias
    )