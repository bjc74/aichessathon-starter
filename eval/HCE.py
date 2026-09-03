import chess
from .tables import PIECE_VALUES, PST

def material_score(board):
    material = sum(
        PIECE_VALUES[piece] * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece in range(chess.PAWN, chess.KING +1)
    )
    return material
def positional_score(board):
    score = 0
    for piece in range(chess.PAWN, chess.KING+1):
        pst = PST[piece]
        for square in board.pieces(piece, chess.WHITE):
                score += pst[chess.square_mirror(square)]
        for square in board.pieces(piece, chess.BLACK):
                score -= pst[square]
    return score
def evaluate(board):
    return material_score(board) + positional_score(board)