import chess
from tables import WHITE_COMBINED, BLACK_COMBINED

def evaluate(board: chess.Board) -> int:
    score = 0
    w_occ = board.occupied_co[chess.WHITE]
    b_occ = board.occupied_co[chess.BLACK]

    # Pawns
    for sq in chess.scan_forward(board.pawns & w_occ):   score += WHITE_COMBINED[1][sq]
    for sq in chess.scan_forward(board.pawns & b_occ):   score -= BLACK_COMBINED[1][sq]

    # Knights
    for sq in chess.scan_forward(board.knights & w_occ): score += WHITE_COMBINED[2][sq]
    for sq in chess.scan_forward(board.knights & b_occ): score -= BLACK_COMBINED[2][sq]

    # Bishops
    for sq in chess.scan_forward(board.bishops & w_occ): score += WHITE_COMBINED[3][sq]
    for sq in chess.scan_forward(board.bishops & b_occ): score -= BLACK_COMBINED[3][sq]

    # Rooks
    for sq in chess.scan_forward(board.rooks & w_occ):   score += WHITE_COMBINED[4][sq]
    for sq in chess.scan_forward(board.rooks & b_occ):   score -= BLACK_COMBINED[4][sq]

    # Queens
    for sq in chess.scan_forward(board.queens & w_occ):  score += WHITE_COMBINED[5][sq]
    for sq in chess.scan_forward(board.queens & b_occ):  score -= BLACK_COMBINED[5][sq]

    # Kings
    for sq in chess.scan_forward(board.kings & w_occ):   score += WHITE_COMBINED[6][sq]
    for sq in chess.scan_forward(board.kings & b_occ):   score -= BLACK_COMBINED[6][sq]

    return score