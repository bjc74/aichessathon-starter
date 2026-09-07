import math
import chess
import chess.polyglot
import time
from tables import PIECE_VALUES
#imported current eval
from eval import evaluate

# Flags for bounds in Transposition Table
EXACT = 0
LB = 1
UB = 2
# Transposition table stores 2^20 values
TT_SIZE = 1_048_576
# Bitwise indexing. Mask is 20 1's, so isolates last 20 of 64-bit Zobrist hash
TT_MASK = TT_SIZE - 1
# Global move counter
current_age = 0
# Cap max ply for killer moves
MAX_PLY = 128
#mate
MATE = 10**6

# Global Transposition Table array
transposition_table = [None] * TT_SIZE
# Global History Table 2D-array used for History Heuristic. history_table[from_square][to_square]
history_table = [[0]*64 for _ in range(64)]
# Global Killer Moves array track primary and secondary killer moves at given ply
killer_moves = [[None, None] for _ in range(MAX_PLY)]
# Global Late Move Reductions lookup table. LMR_TABLE[depth][move_index]
LMR_TABLE = [[0] * 64 for _ in range(64)]
for d in range(1,64):
    for m in range(1,64):
        LMR_TABLE[d][m] = int(0.5 + (math.log(d) * math.log(m)) / 2)

# Custom exception raised when a move runs out of its allocated time
class TimeoutException(Exception):
    pass


# Scoring for Move Ordering (Most Valuable Victim Least Valuable Attacker)
def score_move(board: chess.Board, move: chess.Move, scoring_const: int = 100, priority_move: chess.Move = None, k1: chess.Move =None, k2: chess.Move = None) -> int:
    # Want a prioritised move to be searched first, may be most optimal
    if move == priority_move:
        return 1_000_000_000

    is_promotion = move.promotion is not None

    if board.is_capture(move):
        # Determine relevant pieces from move. With en-passant, 'to' square empty so set pawn
        attacker_piece = board.piece_at(move.from_square).piece_type
        if board.is_en_passant(move):
            victim_piece = chess.PAWN
        else:
            victim_piece = board.piece_at(move.to_square).piece_type

        # .get used to default val to 0 in case given piece not in PIECE_VALUES
        attacker_value, victim_value = PIECE_VALUES.get(attacker_piece, 0), PIECE_VALUES.get(victim_piece, 0)

        # Score offset by 10^8 so captures always ranked above quiet moves
        # scoring_const is subject to piece values. Altenative to this method is 2D lookup table
        mvv_lva = 100_000_000 + (scoring_const * victim_value) - attacker_value

        # Promoting capture scores higher
        if is_promotion and move.promotion == chess.QUEEN:
            return mvv_lva + 100_000_000
        return mvv_lva

    # Quiet promotions rank next highest
    if is_promotion:
        if move.promotion == chess.QUEEN:
            return 90_000_000
        elif move.promotion == chess.KNIGHT:
            return 9_500_000 # Ranks just above primary quiet killer move

    # Check if quiet move is a killer move at this ply
    if move == k1:
        return 9_000_000
    if move == k2:
        return 8_000_000

    # Quiet moves ranked via history heuristic
    return history_table[move.from_square][move.to_square]

# Prevent horizon effect by exploring capture chains until quiet board state
def quiescence_search(board: chess.Board, alpha: float, beta: float, start_time: float, time_limit: float, node_count: list, ply: int) -> float:
    # Check if move time limit exceeded every 2048 nodes
    node_count[0] += 1
    if not (node_count[0] & 2047):
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
        # stand-pat enables 'standing pat', break capture chain to not force captures if not optimal
        stand_pat = evaluate(board)

        # If current state > beta, it cannot be reached so prune
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Delta pruning. If standing pat plus max possible material gain from capture, plus safety margin
        # is still below alpha, capture is hopeless, may be skipped
        BIG_DELTA = 900 # Queen value
        if stand_pat + BIG_DELTA < alpha:
            return alpha

        # Filter out only moves which result in capture
        moves = list(board.generate_legal_captures())

    # Sort moves for optimal pruning
    killer_move_1, killer_move_2 = killer_moves[ply] if ply < MAX_PLY else (None, None)
    # i needed to break ties in sorting when scores are equal
    scored_moves = [(score_move(board, x, k1=killer_move_1, k2=killer_move_2), i, x) for i, x in enumerate(moves)]
    scored_moves.sort(reverse=True)
    moves = [m for _,_, m in scored_moves]

    for move in moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, start_time, time_limit, node_count, ply+1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def negamax(board: chess.Board, depth: int, alpha: float, beta: float, start_time: float, time_limit: float, node_count: list, ply: int, allow_null: bool = True) -> float:
    # Dynamic Contempt: If option to draw, condemn if winning
    if ply > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
        # If eval is pos, draw score neg (bad). If eval neg, draw score 0 (neutral)
        eval_val = evaluate(board) if not board.is_check() else 0
        draw_score = min(0, -int(eval_val*0.5))
        return draw_score

    # Check if move time limit exceeded every 2048 nodes
    node_count[0] += 1
    if not (node_count[0] & 2047):
        if time.time() - start_time > time_limit:
            raise TimeoutException()

    alpha_initial = alpha
    key = chess.polyglot.zobrist_hash(board)
    tt_move = None

    # Check if board state is in transposition table
    idx = key & TT_MASK
    entry = transposition_table[idx]
    if entry is not None and entry['key'] == key:
        tt_move = entry['move']

        # Only used cached score if depth from entry exceeds current depth, otherwise may not be otptimal
        if entry['depth'] >= depth:
            cached_score = entry['score']
            # Re-convert mate distance from being top-relative to bottom-relative
            if cached_score > MATE - 1000:
                cached_score -= ply
            elif cached_score < - MATE + 1000:
                cached_score += ply

            if (entry['flag'] == EXACT) or (entry['flag'] == LB and cached_score >= beta) or (entry['flag'] == UB and cached_score <= alpha):
                return cached_score

    moves = list(board.legal_moves)
    if not moves:
        # MATE - ply incentivises engine to prioritise the move leading to the MATE in less moves, if multiple
        return -(MATE-ply) if board.is_check() else 0.0
    if depth == 0:
        return quiescence_search(board, alpha, beta, start_time, time_limit, node_count, ply)

    # Reverse Futility Pruning: If near leaf node and board state is really good (eval minus a margin is still bigger than beta),
    # then futile to search moves. Position is overwhelmingly winning, prune branch.
    static_eval = evaluate(board) if not board.is_check() else -math.inf
    if depth <= 3 and not board.is_check() and beta < MATE - 1000:
        RFP_margin = 120 * depth
        if static_eval - RFP_margin >= beta:
            return static_eval

    # Null Move Pruning (Simulate giving opponent extra move, large advantage, and search with reduced window + depth.
    # if still beta-cutoff, current board position too strong, prune). R is reduced depth amount.
    R = 2
    # Do not NMP if in check (cannot skip move here...) or if end game and opponent can only move king/pawns
    # Latter to prevent NMP occuring in zugzwang, where skipping move isn't disadvantage and would defeat NMP purpose
    if allow_null and beta < MATE - 1000 and depth >= R + 1 and not board.is_check() and bool(board.occupied_co[board.turn] & ~board.pawns & ~board.kings):
        board.push(chess.Move.null())
        # Reduced depth and window, and allow_null set to False prevents adjacent null moves
        null_score = -negamax(board, depth-1-R, -beta, -beta + 1, start_time, time_limit, node_count, ply+1, allow_null=False)
        board.pop()
        if null_score >= beta:
            return beta

    # sort moves via MVV-LVA for efficient pruning, prioritise move stored in TT
    killer_move_1, killer_move_2 = killer_moves[ply] if ply < MAX_PLY else (None, None)
    scored_moves = [(score_move(board, x, priority_move = tt_move, k1=killer_move_1, k2=killer_move_2), i, x) for i, x in enumerate(moves)]
    scored_moves.sort(reverse=True)
    moves = [m for _,_, m in scored_moves]
    best_score = -math.inf
    best_move = None

    for i, move in enumerate(moves):
        # Futility Pruning, skip non tactical moves that cannot reach alpha (quiet move / move that gains no material cannot
        # pull you out of a deep hole, futile to check, so skip). i > 0 check prevents skipping every move if all are quiet
        if i > 0 and depth <= 2 and not board.is_check() and alpha > -MATE + 1000:
            FP_margin = 200 * depth
            if not board.is_capture(move) and not move.promotion and not board.gives_check(move) and (static_eval + FP_margin <= alpha):
                continue

        board.push(move)
        # Perform principal variation search: With efficient move ordering, first move highly likely to be optimal
        if i == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, node_count, ply+1, allow_null=True)
        # Every other move searched with zero window (need to prove cheaply that move is worse than first, no need for find exact score)
        else:
            # LMR eligibility: Late move, depth >= 3, quiet move, not in check, does not give check
            if i >= 3 and depth >= 3 and not board.is_capture(move) and not move.promotion and not board.is_check() and not board.gives_check(move):
                # If LMR eligible, search later moves with reduced depth from lookup table
                reduction = LMR_TABLE[min(depth,63)][min(i,63)]
                reduced_depth = max(0, depth -1 -reduction)

                score = -negamax(board, reduced_depth, -alpha - 1, -alpha, start_time, time_limit, node_count, ply+1, allow_null=True)

                # If score > alpha, want to look again still with zero window, but with full depth
                do_full_depth_search = score > alpha
            else:
                # If not eligible for LMR, perform PVS using full depth
                do_full_depth_search = True

            if do_full_depth_search:
                # Full depth zero window search
                score = -negamax(board, depth-1, -alpha - 1, -alpha, start_time, time_limit, node_count, ply+1, allow_null=True)

                # Full depth full window re-search if move is promising
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, node_count, ply+1, allow_null=True)

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            if not board.is_capture(move):
                # Reward quiet move that caused beta-cutoff, using depth^2 (bigger depth means more prune)
                history_table[move.from_square][move.to_square] += depth*depth

                # Update killer moves at this ply
                if ply < MAX_PLY and move != killer_moves[ply][0]:
                    killer_moves[ply][1] = killer_moves[ply][0]
                    killer_moves[ply][0] = move

            break

    # Add to transposition table, replace exisiting if new depth larger
    idx = key & TT_MASK
    entry = transposition_table[idx]
    # If collision, replace stale entry
    if entry is None or entry['key'] == key or depth >= entry['depth'] or entry['age'] != current_age:
        # Determine flag
        if best_score <= alpha_initial:
            flag = UB
        elif best_score >= beta:
            flag = LB
        else:
            flag = EXACT

        # Convert MATE score from being bottom-relative to top-relative when storing
        # (Encode the distance of mate from this board state than from depth limit)
        score_to_store = best_score
        if best_score > MATE - 1000:
            score_to_store += ply
        elif best_score < - MATE + 1000:
            score_to_store -= ply

        transposition_table[idx] = {'key': key, 'depth': depth, 'score': score_to_store, 'flag': flag, 'move': best_move, 'age': current_age}

    return best_score

def get_move(fen: str, time_left_ms: int) -> str:
    global history_table, killer_moves, current_age
    current_age += 1

    board = chess.Board(fen)
    # Clear killer moves
    killer_moves = [[None, None] for _ in range(MAX_PLY)]

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return ''

    # Adapt remaining moves dynamically
    remaining_moves = max(10, 50 - board.fullmove_number)
    increment_time = 0.5

    # Determine move time window
    time_limit = ((time_left_ms / 1000) / remaining_moves) + increment_time
    # Never spend more than 80% of remaining time on single move
    time_limit = min(time_limit, (time_left_ms / 1000) * 0.8)
    start_time = time.time()
    previous_depth_time = 0
    # Count number of nodes checked so every 2048 nodes, can check if time limit exceeded
    node_count = [0]

    best_score = -math.inf
    best_move = legal_moves[0]

    # Iterative deepening, to get as deep as possible in given time window
    for depth in range(1, 64):
        # Assume next depth will take 2.5x longer than previous. If remaining time less than this, do not attempt depth
        if depth > 1 and time_limit - (time.time() - start_time) < previous_depth_time * 2.5:
            break

        # TimeoutException will be thrown if time limit exceeded hence try except block
        try:
            # Find the best move determined at given depth
            current_best_score = -math.inf
            current_best_move = None
            alpha = -math.inf
            beta = math.inf
            depth_start_time = time.time()

            # Prioritise searching best move determined from previous depth first, likely to also be best at this depth
            killer_move_1, killer_move_2 = killer_moves[0]
            scored_moves = [(score_move(board, x, priority_move =best_move, k1=killer_move_1, k2=killer_move_2), i, x) for i, x in enumerate(legal_moves)]
            scored_moves.sort(reverse=True)
            ordered_moves = [m for _, _, m in scored_moves]

            for i, move in enumerate(ordered_moves):
                board.push(move)
                # Principal variation search, just like in negamax function
                if i == 0:
                    score = -negamax(board, depth-1, -beta, -alpha, start_time, time_limit, node_count, 1)
                else:
                    score = -negamax(board, depth-1, -alpha-1, -alpha, start_time, time_limit, node_count, 1)
                    if alpha < score < beta:
                        score = -negamax(board, depth-1, -beta, -alpha, start_time, time_limit, node_count, 1)
                board.pop()

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                alpha = max(alpha, score)

            # Only overwrite best_move if a best move from this iteration is deduced
            if current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score

            # Find the time taken at this depth, to determine if enough time for a deeper search    
            previous_depth_time = time.time() - depth_start_time

        except TimeoutException:
            break

    # Age history table values from older iterations to reward newer scores from deeper searches
    for i in range(64):
        for j in range(64):
            history_table[i][j] //= 2

    return best_move.uci()