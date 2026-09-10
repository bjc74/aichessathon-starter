import math
import time
import chess
import agent


POSITIONS = [
    (
        "starting_position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "e4_regression",
        "r2q3r/p2kb3/2p1b1Q1/2p4p/3p2p1/6P1/PPNPPP2/R1B2RK1 w - - 2 18",
    ),
    (
        "castling_regression",
        "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/3p1P2/3B1N2/PPP1N1PP/R1BQK2R w KQkq - 4 9",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "rook_pawn_endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ),
    (
        "tactical_development",
        "rnbq1k1r/pp1Pbppp/2p2n2/8/2B5/8/PPP1NPPP/RNBQK2R b KQ - 1 8",
    ),
    (
        "quiet_middlegame",
        "r4rk1/1pp1qppp/p1np1n2/8/2B1P3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 10",
    ),
    (
        "queens_gambit",
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    ),
    (
        "promotion_race",
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
    ),
    (
        "queen_endgame",
        "6k1/5ppp/8/8/8/8/5PPP/6KQ w - - 0 1",
    ),
]


MAX_DEPTH = 6


def reset_engine_state():
    agent.transposition_table = [None] * len(agent.transposition_table)
    agent.history_table = [[0 for _ in range(64)] for _ in range(64)]
    agent.killer_moves = [[None, None] for _ in range(agent.MAX_PLY)]
    agent.current_age += 1


def search_root(board, depth):
    agent.initialise_nnue(board)
    start_time = time.time()

    # Effectively disable timeout for fixed-depth testing
    time_limit = 10_000.0
    node_count = [0]

    alpha = -math.inf
    beta = math.inf

    best_score = -math.inf
    best_move = None

    legal_moves = list(board.legal_moves)

    scored_moves = [
        (
            agent.score_move(
                board,
                move,
                priority_move=None,
                k1=None,
                k2=None,
            ),
            i,
            move,
        )
        for i, move in enumerate(legal_moves)
    ]

    scored_moves.sort(reverse=True)
    ordered_moves = [move for _, _, move in scored_moves]

    for i, move in enumerate(ordered_moves):
        agent.push_nnue(board, move)

        if i == 0:
            score = -agent.negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                start_time,
                time_limit,
                node_count,
                1,
            )
        else:
            score = -agent.negamax(
                board,
                depth - 1,
                -alpha - 1,
                -alpha,
                start_time,
                time_limit,
                node_count,
                1,
            )

            if alpha < score < beta:
                score = -agent.negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    start_time,
                    time_limit,
                    node_count,
                    1,
                )

        agent.pop_nnue(board)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    elapsed = time.time() - start_time

    return best_move, best_score, node_count[0], elapsed


def main():
    total_nodes_all_depths = 0
    total_time_all_depths = 0.0

    depth_totals = {
        depth: 0 for depth in range(1, MAX_DEPTH + 1)
    }

    for name, fen in POSITIONS:
        print()
        print("=" * 70)
        print(name)
        print("=" * 70)

        reset_engine_state()

        board = chess.Board(fen)

        for depth in range(1, MAX_DEPTH + 1):
            move, score, nodes, elapsed = search_root(board, depth)

            nps = nodes / elapsed if elapsed > 0 else 0

            total_nodes_all_depths += nodes
            total_time_all_depths += elapsed
            depth_totals[depth] += nodes

            print(
                f"D{depth}: "
                f"move={move.uci() if move else None}, "
                f"score={score}, "
                f"nodes={nodes:,}, "
                f"time={elapsed:.3f}s, "
                f"NPS={nps:,.0f}"
            )

    print()
    print("=" * 70)
    print("TOTALS")
    print("=" * 70)

    for depth in range(1, MAX_DEPTH + 1):
        print(
            f"D{depth} total nodes: "
            f"{depth_totals[depth]:,}"
        )

    print(
        f"\nAll-depth total nodes: "
        f"{total_nodes_all_depths:,}"
    )

    print(
        f"Total wall time: "
        f"{total_time_all_depths:.3f}s"
    )

    overall_nps = (
        total_nodes_all_depths / total_time_all_depths
        if total_time_all_depths > 0
        else 0
    )

    print(
        f"Overall NPS: "
        f"{overall_nps:,.0f}"
    )


if __name__ == "__main__":
    main()