import chess

from nnue_inference import evaluate_nnue


# How much explicit material value to add.
# NNUE already appears to have learned roughly 15-25% of material value,
# so 0.8 is a sensible first experiment.
MATERIAL_WEIGHT = 0.8


MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def material_score(board):
    """
    Material score from White's perspective.
    Positive = White has more material.
    Negative = Black has more material.
    """

    score = 0

    for piece_type, value in MATERIAL_VALUES.items():

        white_count = chess.popcount(
            board.pieces_mask(
                piece_type,
                chess.WHITE
            )
        )

        black_count = chess.popcount(
            board.pieces_mask(
                piece_type,
                chess.BLACK
            )
        )

        score += value * (
            white_count - black_count
        )

    return score


def evaluate_hybrid_nnue(board):
    """
    Hybrid evaluator from side-to-move perspective.

    NNUE score
        +
    0.8 * explicit material score
    """

    nnue_score = evaluate_nnue(board)

    material = material_score(board)

    # material_score() is White-perspective,
    # while evaluate_nnue() is side-to-move perspective.
    if board.turn == chess.BLACK:
        material = -material

    score = (
        nnue_score
        + MATERIAL_WEIGHT * material
    )

    return int(score)


if __name__ == "__main__":

    board = chess.Board()

    print(
        "Starting position:",
        evaluate_hybrid_nnue(board)
    )