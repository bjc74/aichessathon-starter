import chess
from tables import WHITE_COMBINED, BLACK_COMBINED


BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 12
ISOLATED_PAWN_PENALTY = 10

# Indexed by how far the pawn has advanced.
PASSED_PAWN_BONUS = [0, 0, 5, 10, 20, 35, 60, 0]

# For each file, contains the two neighbouring files.
ADJACENT_FILE_MASKS = [0] * 8

for file in range(8):

    mask = 0

    if file > 0:
        mask |= chess.BB_FILES[file - 1]

    if file < 7:
        mask |= chess.BB_FILES[file + 1]

    ADJACENT_FILE_MASKS[file] = mask


# For each square, squares on the same/adjacent files
# that contain an enemy pawn capable of stopping it
# from being a passed pawn.
WHITE_PASSED_MASKS = [0] * 64
BLACK_PASSED_MASKS = [0] * 64

for sq in range(64):

    file = chess.square_file(sq)
    rank = chess.square_rank(sq)

    files = range(
        max(0, file - 1),
        min(7, file + 1) + 1
    )

    white_mask = 0
    black_mask = 0

    # Squares ahead of a White pawn
    for r in range(rank + 1, 8):
        for f in files:
            white_mask |= chess.BB_SQUARES[chess.square(f, r)]

    # Squares ahead of a Black pawn
    for r in range(0, rank):
        for f in files:
            black_mask |= chess.BB_SQUARES[chess.square(f, r)]

    WHITE_PASSED_MASKS[sq] = white_mask
    BLACK_PASSED_MASKS[sq] = black_mask
def evaluate(board: chess.Board) -> int:

    score = 0

    w_occ = board.occupied_co[chess.WHITE]
    b_occ = board.occupied_co[chess.BLACK]

    white_pawns = board.pawns & w_occ
    black_pawns = board.pawns & b_occ

    white_bishops = board.bishops & w_occ
    black_bishops = board.bishops & b_occ

    # ---------------------------------------------------------
    # Material + PST
    # ---------------------------------------------------------

    for sq in chess.scan_forward(white_pawns):
        score += WHITE_COMBINED[chess.PAWN][sq]

    for sq in chess.scan_forward(black_pawns):
        score -= BLACK_COMBINED[chess.PAWN][sq]

    for sq in chess.scan_forward(board.knights & w_occ):
        score += WHITE_COMBINED[chess.KNIGHT][sq]

    for sq in chess.scan_forward(board.knights & b_occ):
        score -= BLACK_COMBINED[chess.KNIGHT][sq]

    for sq in chess.scan_forward(white_bishops):
        score += WHITE_COMBINED[chess.BISHOP][sq]

    for sq in chess.scan_forward(black_bishops):
        score -= BLACK_COMBINED[chess.BISHOP][sq]

    for sq in chess.scan_forward(board.rooks & w_occ):
        score += WHITE_COMBINED[chess.ROOK][sq]

    for sq in chess.scan_forward(board.rooks & b_occ):
        score -= BLACK_COMBINED[chess.ROOK][sq]

    for sq in chess.scan_forward(board.queens & w_occ):
        score += WHITE_COMBINED[chess.QUEEN][sq]

    for sq in chess.scan_forward(board.queens & b_occ):
        score -= BLACK_COMBINED[chess.QUEEN][sq]

    for sq in chess.scan_forward(board.kings & w_occ):
        score += WHITE_COMBINED[chess.KING][sq]

    for sq in chess.scan_forward(board.kings & b_occ):
        score -= BLACK_COMBINED[chess.KING][sq]

    # ---------------------------------------------------------
    # Bishop pair
    # ---------------------------------------------------------

    if chess.popcount(white_bishops) >= 2:
        score += BISHOP_PAIR_BONUS

    if chess.popcount(black_bishops) >= 2:
        score -= BISHOP_PAIR_BONUS

    # ---------------------------------------------------------
    # Pawn structure: doubled + isolated
    # ---------------------------------------------------------

    for file in range(8):

        file_mask = chess.BB_FILES[file]

        white_on_file = white_pawns & file_mask
        black_on_file = black_pawns & file_mask

        white_count = chess.popcount(white_on_file)
        black_count = chess.popcount(black_on_file)

        # One pawn is normal; each additional pawn is doubled.
        if white_count > 1:
            score -= DOUBLED_PAWN_PENALTY * (white_count - 1)

        if black_count > 1:
            score += DOUBLED_PAWN_PENALTY * (black_count - 1)

        # If a file contains pawns and there are no friendly pawns
        # on either adjacent file, those pawns are isolated.
        if white_on_file and not (
            white_pawns & ADJACENT_FILE_MASKS[file]
        ):
            score -= ISOLATED_PAWN_PENALTY * white_count

        if black_on_file and not (
            black_pawns & ADJACENT_FILE_MASKS[file]
        ):
            score += ISOLATED_PAWN_PENALTY * black_count

    # ---------------------------------------------------------
    # Passed pawns
    # ---------------------------------------------------------

    for sq in chess.scan_forward(white_pawns):

        if not (black_pawns & WHITE_PASSED_MASKS[sq]):

            rank = chess.square_rank(sq)

            score += PASSED_PAWN_BONUS[rank]

    for sq in chess.scan_forward(black_pawns):

        if not (white_pawns & BLACK_PASSED_MASKS[sq]):

            rank = chess.square_rank(sq)

            # Black starts on rank index 6 and moves toward 0.
            progress = 7 - rank

            score -= PASSED_PAWN_BONUS[progress]

    return score