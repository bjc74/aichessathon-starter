import chess
from hce import evaluate as hce_evaluate
def evaluate(board, mobility=0):
    score = hce_evaluate(board)
    # Negamax wants evaluation from side-to-move perspective
    if board.turn == chess.BLACK:
        score = -score
    return score