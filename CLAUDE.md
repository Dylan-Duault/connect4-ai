# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance AlphaZero implementation for Connect 4 using PyTorch with CUDA 13 support. The system implements the complete AlphaZero algorithm including neural network training, Monte Carlo Tree Search (MCTS), and self-play data generation.

## Key Commands

### Environment Setup
```bash
# Activate virtual environment (always required)
source venv/bin/activate

# Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Testing Individual Components
```bash
cd src
python game.py          # Test game engine
python neural_network.py # Test neural network with GPU
python mcts.py          # Test MCTS implementation
python self_play.py     # Test self-play system
```

### Training
```bash
# Start new training
python train.py --games-per-iteration 50 --mcts-simulations 400 --iterations 30

# Resume from checkpoint
python train.py --resume models/checkpoint_iter_0010.pth --iterations 50

# Progressive training (recommended approach)
python train.py --games-per-iteration 50 --mcts-simulations 200 --iterations 30  # Fast start
python train.py --resume models/checkpoint_iter_0030.pth --mcts-simulations 600 --iterations 80  # Medium
python train.py --resume models/checkpoint_iter_0080.pth --mcts-simulations 1200 --iterations 150  # Full strength
```

### Playing Against AI
```bash
# Command line interface
python play.py --model models/final_model.pth --ai-analysis

# Graphical interface
python gui_game.py
```

## Architecture Overview

### Core Components Flow
1. **GameState & GameBoard** (`src/game.py`): Core game logic with efficient numpy arrays and PyTorch tensor conversion
2. **AlphaZeroNetwork** (`src/neural_network.py`): Dual-head CNN (policy + value) with residual blocks, ~9.8M parameters
3. **MCTS** (`src/mcts.py`): Monte Carlo Tree Search guided by neural network predictions
4. **Self-Play** (`src/self_play.py`): Generates training data through AI vs AI games
5. **Training Loop** (`train.py`): Orchestrates self-play → neural network training → checkpointing cycle

### Key Architectural Patterns

**Tensor Pipeline**: Game states flow through: `numpy.ndarray` (game logic) → `torch.Tensor` (neural network) → `numpy.ndarray` (MCTS) in a efficient cycle.

**Training Data Flow**: Self-play generates `TrainingExample` objects containing (board_state, action_probabilities, game_outcome) which are batched for neural network training.

**MCTS Integration**: The `MCTSNode` tree structure stores visit counts and values, with neural network providing prior probabilities and leaf evaluation.

**Checkpoint System**: Models save both `network_state_dict` and training metadata, with separate pickle files for training data to handle PyTorch 2.6 compatibility.

## Important Implementation Details

### Neural Network Architecture
- Input: 3-channel tensor (current player pieces, opponent pieces, player indicator)
- Architecture: Initial conv → 8 residual blocks → dual heads (policy/value)
- Policy head outputs 7 values (one per column)
- Value head outputs single tanh value (-1 to +1)

### MCTS Implementation
- UCB formula: `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- Temperature scheduling: High temperature (1.0) early game for exploration, low (0.1) late game
- Batch neural network calls for efficiency during tree expansion

### Self-Play Training
- Games use MCTS with current neural network for move selection
- Training examples store position, MCTS probabilities, and final game outcome
- Outcome assignment: +1 for win, -1 for loss, 0 for draw from each player's perspective
- Rolling buffer of 100k examples prevents overfitting to recent games

### Performance Optimization
- GPU acceleration throughout the pipeline
- Threaded AI moves in GUI to prevent interface freezing
- Progressive MCTS simulation scaling (200 → 400 → 800+ as model improves)
- Efficient tensor operations with proper device placement

## Training Progression Expectations

- **Iterations 1-10**: Slow games (~60s each), untrained model provides poor MCTS guidance
- **Iterations 11-50**: Accelerating (~20s per game), tactical play emerges
- **Iterations 51+**: Fast games (~10s), strategic understanding develops
- **Win rate convergence**: Should approach 50/50 Red/Yellow with increasing draw rate as play strengthens

## Model Loading Compatibility

The codebase handles PyTorch 2.6 weight loading changes with fallback:
```python
try:
    checkpoint = torch.load(filepath, weights_only=True)
except Exception:
    checkpoint = torch.load(filepath, weights_only=False)
```

Models are saved with `_use_new_zipfile_serialization=False` for backward compatibility.

## Monitoring and Debugging

- Training logs in `logs/training_log.json` with detailed metrics per iteration
- Automatic plot generation showing loss curves and win rates in `logs/training_progress.png`
- GPU memory monitoring via `nvidia-smi` if performance issues arise
- Win rate variance tracking for early stopping detection (<0.001 indicates convergence)