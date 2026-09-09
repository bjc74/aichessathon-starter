import chess
from tables import WHITE_COMBINED, BLACK_COMBINED, KING_MIDDLEGAME, KING_ENDGAME


BISHOP_PAIR_BONUS = 30
DOUBLED_PAWN_PENALTY = 12
ISOLATED_PAWN_PENALTY = 10
ROOK_SEMI_OPEN_BONUS = 10
ROOK_OPEN_BONUS = 20
# Indexed by how far the pawn has advanced.
PASSED_PAWN_BONUS = [0, 0, 5, 10, 20, 35, 60, 0]

# For each file, contains the two neighbouring files.
ADJACENT_FILE_MASKS = [0] * 8
KING_SHIELD_PENALTY = 12

WHITE_KING_SHIELD = [0] * 64
BLACK_KING_SHIELD = [0] * 64

for sq in range(64):

    file = chess.square_file(sq)
    rank = chess.square_rank(sq)

    # Squares one rank in front of White king
    if rank < 7:
        for f in range(max(0, file - 1), min(7, file + 1) + 1):
            WHITE_KING_SHIELD[sq] |= chess.BB_SQUARES[
                chess.square(f, rank + 1)
            ]

    # Squares one rank in front of Black king
    if rank > 0:
        for f in range(max(0, file - 1), min(7, file + 1) + 1):
            BLACK_KING_SHIELD[sq] |= chess.BB_SQUARES[
                chess.square(f, rank - 1)
            ]

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
MAX_PHASE = 24

def evaluate(board: chess.Board) -> int:

    score = 0

    w_occ = board.occupied_co[chess.WHITE]
    b_occ = board.occupied_co[chess.BLACK]

    white_pawns = board.pawns & w_occ
    black_pawns = board.pawns & b_occ

    white_bishops = board.bishops & w_occ
    black_bishops = board.bishops & b_occ

    white_rooks = board.rooks & w_occ
    black_rooks = board.rooks & b_occ   

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

    for sq in chess.scan_forward(white_rooks):
        score += WHITE_COMBINED[chess.ROOK][sq]

    for sq in chess.scan_forward(black_rooks):
        score -= BLACK_COMBINED[chess.ROOK][sq]

    for sq in chess.scan_forward(board.queens & w_occ):
        score += WHITE_COMBINED[chess.QUEEN][sq]

    for sq in chess.scan_forward(board.queens & b_occ):
        score -= BLACK_COMBINED[chess.QUEEN][sq]

    for sq in chess.scan_forward(board.kings & w_occ):
        score += WHITE_COMBINED[chess.KING][sq]

    for sq in chess.scan_forward(board.kings & b_occ):
        score -= BLACK_COMBINED[chess.KING][sq]
    # Game phase: 24 at the start, 0 in a pure pawn ending
    phase = (
        chess.popcount(board.knights)
        + chess.popcount(board.bishops)
        + 2 * chess.popcount(board.rooks)
        + 4 * chess.popcount(board.queens)
    )

    phase = min(phase, 24)
    
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    # Basic king pawn shield
    w_shield_mask = WHITE_KING_SHIELD[white_king]
    b_shield_mask = BLACK_KING_SHIELD[black_king]

    w_expected = chess.popcount(w_shield_mask)
    b_expected = chess.popcount(b_shield_mask)

    w_shield = chess.popcount(white_pawns & w_shield_mask)
    b_shield = chess.popcount(black_pawns & b_shield_mask)

    w_missing = w_expected - w_shield
    b_missing = b_expected - b_shield

    # Strong in middlegame, fades to zero in endgame
    score -= (
        w_missing * KING_SHIELD_PENALTY * phase
    ) // 24

    score += (
        b_missing * KING_SHIELD_PENALTY * phase
    ) // 24
    # How far towards an endgame we are
    endgame_weight = 24 - phase
    # Current score already contains the middlegame king PST,
    # so gradually add the difference between EG and MG.
    w_sq = white_king ^ 56
    b_sq = black_king

    score += (
        endgame_weight
        * (KING_ENDGAME[w_sq] - KING_MIDDLEGAME[w_sq])
        //24
    )

    score -= (
        endgame_weight
        * (KING_ENDGAME[b_sq] - KING_MIDDLEGAME[b_sq])
        // 24
    )

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
        white_rooks_on_file = white_rooks & file_mask
        black_rooks_on_file = black_rooks & file_mask

        # White rook
        if white_rooks_on_file and not white_on_file:

            if not black_on_file:
                score += ROOK_OPEN_BONUS * chess.popcount(white_rooks_on_file)
            else:
                score += ROOK_SEMI_OPEN_BONUS * chess.popcount(white_rooks_on_file)


        # Black rook
        if black_rooks_on_file and not black_on_file:

            if not white_on_file:
                score -= ROOK_OPEN_BONUS * chess.popcount(black_rooks_on_file)
            else:
                score -= ROOK_SEMI_OPEN_BONUS * chess.popcount(black_rooks_on_file)

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