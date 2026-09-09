import time

import agent


FEN = "r2q3r/p2kb3/2p1b1Q1/2p4p/3p2p1/6P1/PPNPPP2/R1B2RK1 w - - 2 18"
TIME_LEFT_MS = 86_201


agent.transposition_table[:] = [None] * len(agent.transposition_table)
agent.history_table = [[0] * 64 for _ in range(64)]
agent.killer_moves = [[None, None] for _ in range(agent.MAX_PLY)]

# Correct counter
agent.lmr = 0


print("=" * 60)
print("REAL GET_MOVE TEST — FIXED LMR")
print("=" * 60)

start = time.perf_counter()

move = agent.get_move(FEN, TIME_LEFT_MS)

elapsed = time.perf_counter() - start


print("\nRESULT")
print(f"Move: {move}")
print(f"Wall time: {elapsed:.3f}s")
print(f"LMR searches: {agent.lmr:,}")