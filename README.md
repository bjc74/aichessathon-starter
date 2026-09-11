# AI Chessathon Chess Engine

A chess engine developed for the 2026 AI Chessathon, built on the official competition starter repository.

The final engine combines a modern alpha-beta search stack with a compact NNUE-style neural evaluator, incremental inference, transposition tables, move-ordering heuristics and Syzygy endgame tablebases.

## Engine Architecture

### Search

The engine uses iterative-deepening negamax with alpha-beta pruning and includes:

- Principal Variation Search (PVS)
- Quiescence search
- Transposition tables
- MVV-LVA capture ordering
- Killer moves and history heuristics
- Late Move Reductions (LMR)
- Null Move Pruning
- Futility and reverse-futility pruning
- Aspiration windows
- Repetition and fifty-move handling
- Dynamic time management
- Root-level Syzygy tablebase lookup

The search was developed and tuned using deterministic node-count benchmarks and automated engine-vs-engine arenas.

## Evaluation

The submitted engine uses a hybrid neural/material evaluation function.

### NNUE-style evaluator

Network architecture:

`773 → 256 → 32 → 1`

Input features consist of:

- 768 piece-square occupancy features
- 4 castling-right features
- 1 side-to-move feature

The model was trained on approximately **1 million Stockfish-labelled chess positions**.

The neural score is combined with an explicit material term to improve tactical robustness, after testing showed that the pure learned evaluator systematically undervalued material losses.

Earlier versions of the engine used a handcrafted evaluator containing:

- Material values
- Piece-square tables
- Bishop-pair bonuses
- Doubled-pawn penalties
- Isolated-pawn penalties
- Passed-pawn bonuses

## Incremental NNUE

To make neural evaluation practical inside the search tree, the first NNUE layer is maintained incrementally rather than recomputed from the full board at every leaf.

The accumulator is updated as moves are pushed and popped and supports:

- Normal moves
- Captures
- Castling
- En passant
- Promotions
- Promotion captures
- Castling-right changes
- Null moves

The remaining network layers are evaluated from the stored accumulator, substantially reducing per-node evaluation overhead.

## Testing and Benchmarking

Development included:

- Fixed-depth search benchmarks
- Deterministic node-count regression tests
- Full-window vs aspiration-window A/B tests
- Engine-vs-engine arena testing
- Held-out NNUE validation
- Material-sensitivity experiments
- Incremental-accumulator equivalence tests
- Special-move correctness tests
- Evaluator latency microbenchmarks
- Push/evaluate/pop node benchmarks
- Syzygy performance A/B testing
- Competition-package smoke tests

The final implementation was tested across thousands of generated positions to ensure incremental evaluator state remained consistent with full recomputation.

## Repository Structure

The competition submission is centred around:

```text
agent.py
tables.py
incremental_nnue.py
nnue_inference.py
nnue.py
weights/
    best_nnue.pth
    best_nnue.onnx
syzygy/
```

The repository also contains the official Chessathon harness and baseline agents used for local testing.

## Running Locally

Install the project:

```bash
git clone https://github.com/bjc74/aichessathon-starter
cd aichessathon-starter
make setup
```

Run a game against a baseline:

```bash
make play
```

Run a local arena:

```bash
make arena
```

Start from a custom position:

```bash
make play FEN="<fen>"
```

A specific opponent can also be selected through the harness:

```bash
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
```

## Packaging

Build a competition submission with:

```bash
make zip
```

This produces `submission.zip` in the format expected by the Chessathon platform.

## Competition

The repository is based on the official [AI Chessathon](https://aichessathon.com) starter project.

Competition rules and submission requirements are documented at:

https://aichessathon.com/docs