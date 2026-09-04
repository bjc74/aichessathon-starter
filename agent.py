import math
import chess
import time

# Custom exception raised when a move runs out of its allocated time
class TimeoutException(Exception):
    pass

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

# Prevent horizon effect by exploring capture chains until quiet board state
def quiescence_search(board: chess.Board, alpha: float, beta: float, start_time: float, time_limit: float, node_count: list, ply: int) -> float:
    # Check if move time limit exceeded every 2048 nodes
    node_count[0] += 1
    if not node_count[0] % 2048:
        if time.time() - start_time > time_limit:
            raise TimeoutException()

    # If in check, cannot only look at captures. Must look at all legal moves
    if board.is_check():
        moves = list(board.legal_moves)
        # If no available moves then checkmate
        if not moves:
            return -(MATE-ply)
    else:
        # Find current board state (can be mid capture chain)
        # Mobility set to zero since generating every legal move for every leaf node will bottleneck
        stand_pat = evaluate(board, 0)

        # If current state > beta, it cannot be reached so prune
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Filter out only moves which result in capture
        moves = [move for move in board.legal_moves if board.is_capture(move)]

    # Sort moves for optimal pruning
    moves.sort(key = lambda x: score_move(board, x), reverse = True)

    # stand-pat enables 'standing pat', break capture chain to not force captures if not optimal
    for move in moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, start_time, time_limit, node_count, ply+1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

# Scoring for Move Ordering (Most Valuable Victim Least Valuable Attacker)
def score_move(board: chess.Board, move: chess.Move, scoring_const: int = 100, priority_move: chess.Move = None) -> int:
    # Want a prioritised move to be searched first, may be most optimal
    if move == priority_move:
        return math.inf

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

def negamax(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float, node_count: list, ply: int) -> float:
    # Check if move time limit exceeded every 2048 nodes
    node_count[0] += 1
    if not node_count[0] % 2048:
        if time.time() - start_time > time_limit:
            raise TimeoutException()

    moves = list(board.legal_moves)
    if not moves:
        # MATE - ply incentivises engine to prioritise the move leading to the MATE in less moves, if multiple
        return -(MATE-ply) if board.is_check() else 0.0
    if depth == 0:
        return quiescence_search(board, alpha, beta, start_time, time_limit, node_count, ply)

    # sort moves via MVV-LVA for efficient pruning
    moves.sort(key = lambda x: score_move(board, x), reverse = True)
    best = -math.inf
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, node_count, ply+1)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best

def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ''

    # Can implement more dynamic approach for remaining moves, currently assumes fixed 30
    remaining_moves = 30
    increment_time = 0.5

    # Determine move time window
    time_limit = ((time_left_ms / 1000) / remaining_moves) + increment_time
    start_time = time.time()
    # Count number of nodes checked so every 2048 nodes, can check if time limit exceeded
    node_count = [0]

    best_score = -math.inf
    best_move = legal_moves[0]

    # Iterative deepening, to get as deep as possible in given time window
    for depth in range(1, 64):
        # If 40% time budget used do not risk searching deeper, likely to exceed limit
        if time.time() - start_time > (time_limit * 0.4):
            break

        # TimeoutException will be thrown if time limit exceeded hence try except block
        try:
            # Find the best move determined at given depth
            current_best_score = -math.inf
            current_best_move = None
            alpha = -math.inf
            beta = math.inf

            # Prioritise searching best move determined from previous depth first, likely to also be best at this depth
            ordered_moves = sorted(legal_moves, key = lambda x: score_move(board, x, priority_move=best_move), reverse = True)
            for move in ordered_moves:
                board.push(move)
                score = -negamax(board, depth-1, -beta, -alpha, start_time, time_limit, node_count, 1)
                board.pop()

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                alpha = max(alpha, score)

            best_move = current_best_move
            best_score = current_best_score

        except TimeoutException:
            break

    return best_move.uci()