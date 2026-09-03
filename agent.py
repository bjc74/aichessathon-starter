"""The submission entrypoint. The platform imports this file and calls get_move."""
import math
import random
import chess
# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.
'''
Assign values to each piece. In future models we can improve on this by adding extra points to 
things like connected rooks, bishop pairs, knights on outputs, pawn chains etc, or take away points
for things like knights with no squares to move, trapped bishops etc
'''

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
MOBILITY_WEIGHT = 4.0
MATE = 10**6

def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUES.items()
    )
    return material + MOBILITY_WEIGHT * mobility

# Scoring for Move Ordering (Most Valuable Victim Least Valuable Attacker)
def MVVLVA_score(board: chess.Board, move: chess.Move, scoring_const: int = 100) -> int:
    if not board.is_capture(move):
        return 0

    # Determine relevant pieces from move. With en-passant, 'to' square empty so set pawn
    attacker_piece = board.piece_at(move.from_square).piece_type
    if board.is_en_passant(move):
        victim_piece = chess.PAWN
    else:
        victim_piece = board.piece_at(move.to_square).piece_type

    # .get used to default val to 0 in case given piece not in PIECE_VALUES
    attacker_value, victim_value = PIECE_VALUES.get(attacker_piece, 0), PIECE_VALUES.get(victim_piece, 0)
    # scoring_const is subject to piece values. Altenative to this method is 2D lookup table
    return (scoring_const * victim_value) - attacker_value

def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        return evaluate(board, len(moves))

    # sort moves via MVV-LVA for efficient pruning
    moves.sort(key = lambda x: MVVLVA_score(board, x), reverse = True)
    best = -math.inf
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


#printing a basic material score, from the perspective of whoever is about to move. 
#Idk if you guys have played chess much, but the bigger the plus the bigger the adv
#e.g. +900 means up an entire queens worth of material
def material_score(board):
    side = board.turn
    material = sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUES.items()
    )
    return material
def positional_score(board):
    side = board.turn

def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game.

    print() is safe. Your stdout is redirected away from the protocol stream, discarded
    during rated games and shown back to you in the validation log.
    """
    board = chess.Board(fen)
    best_score = -MATE
    best_moves = []
    for move in list(board.legal_moves):
            board.push(move)
            score = MATE if board.is_checkmate() else -material_score(board)
            # score = -negamax(board, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
    return random.choice(best_moves).uci()
    print(material_score(board), flush=True)#flush = true needed to print

    # Everything from here down is yours to replace. baselines/greedy searches one ply,
    # baselines/minimax searches two. Neither is strong. Reading them is the fastest way
    # to see the shape of a search, and beating them is the first real milestone.
    return random.choice(list(board.legal_moves)).uci()