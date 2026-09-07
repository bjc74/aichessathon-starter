import math
import time
import chess

import agent


FEN = "r2q3r/p2kb3/2p1b1Q1/2p4p/3p2p1/6P1/PPNPPP2/R1B2RK1 w - - 2 18"


def best_move_at_depth(fen, depth):
    board = chess.Board(fen)

    start_time = time.time()
    time_limit = 9999
    node_count = [0]

    best_move = None
    best_score = -math.inf

    alpha = -math.inf
    beta = math.inf

    for move in board.legal_moves:
        board.push(move)

        score = -agent.negamax(
            board,
            depth - 1,
            -beta,
            -alpha,
            start_time,
            time_limit,
            node_count,
            1
        )

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    return best_move, best_score, node_count[0]


for depth in range(1, 7):
    agent.transposition_table[:] = [None] * len(agent.transposition_table)

    start = time.perf_counter()

    move, score, nodes = best_move_at_depth(FEN, depth)

    elapsed = time.perf_counter() - start

    print(
        f"Depth {depth}: "
        f"move={move}, "
        f"score={score}, "
        f"nodes={nodes}, "
        f"time={elapsed:.3f}s, "
        f"NPS={nodes / elapsed:,.0f}"
    )