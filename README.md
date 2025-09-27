# 🔴 AlphaZero Connect 4 AI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance implementation of DeepMind's AlphaZero algorithm for Connect 4, built with PyTorch and optimized for GPU training. This implementation features a neural network with dual heads (policy and value), Monte Carlo Tree Search (MCTS), and self-play training.

## 🚀 Features

### Core AlphaZero Components
- **Neural Network**: Dual-head CNN with residual blocks (9.8M parameters)
- **MCTS**: GPU-accelerated Monte Carlo Tree Search
- **Self-Play**: High-throughput training data generation
- **Training Loop**: Complete AlphaZero training with monitoring

### Performance Optimizations
- **GPU Acceleration**: Full CUDA 13 support with PyTorch 2.10
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: Optimized data structures
- **Multi-threading**: Parallel game generation

## 📋 Requirements

### System Requirements
- **Python**: 3.11 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended)
- **Memory**: 4GB+ GPU memory, 8GB+ system RAM
- **OS**: Linux, macOS, or Windows

### Dependencies
- PyTorch 2.0+
- NumPy
- Matplotlib (for visualization)
- tqdm (for progress bars)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Dylan-Duault/connect4-ai.git
cd connect4-ai
```

### 2. Set up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Option 1: Install from requirements.txt
pip install -r requirements.txt

# Option 2: Manual installation
pip install torch torchvision numpy matplotlib tqdm
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> **Note**: GPU support is optional but highly recommended for faster training. The system will automatically fall back to CPU if CUDA is not available.

## 🎯 Quick Start

### Test the Implementation
```bash
# Test game engine
cd src && python game.py

# Test neural network
python neural_network.py

# Test MCTS
python mcts.py

# Test self-play
python self_play.py
```

### Start Training
```bash
# Basic training (50 games per iteration, 800 MCTS simulations)
python train.py

# High-performance training
python train.py --games-per-iteration 100 --mcts-simulations 1600 --iterations 200

# Quick training for testing
python train.py --games-per-iteration 20 --mcts-simulations 200 --iterations 10
```

### Play Against AI
```bash
# Play against trained model
python play.py --model models/final_model.pth

# Play with AI analysis shown
python play.py --model models/final_model.pth --ai-analysis

# Adjust AI strength
python play.py --model models/final_model.pth --ai-simulations 400  # Weaker
python play.py --model models/final_model.pth --ai-simulations 1600 # Stronger
```

## ⚙️ Training Configuration

### Command Line Options
```bash
python train.py --help
```

Key parameters:
- `--iterations`: Number of training iterations (default: 100)
- `--games-per-iteration`: Self-play games per iteration (default: 50)
- `--mcts-simulations`: MCTS simulations per move (default: 800)
- `--training-epochs`: Neural network epochs per iteration (default: 10)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--checkpoint-freq`: Save frequency (default: 10)

### Performance Tuning

**For High-End GPUs (RTX 4080+, RTX 5070+):**
```bash
# Optimal settings for powerful GPUs
python train.py \
  --games-per-iteration 100 \
  --mcts-simulations 1200 \
  --batch-size 64 \
  --training-epochs 15 \
  --iterations 150
```

**For Mid-Range GPUs (RTX 3060, RTX 4060):**
```bash
# Balanced settings
python train.py \
  --games-per-iteration 50 \
  --mcts-simulations 800 \
  --batch-size 32 \
  --training-epochs 10 \
  --iterations 100
```

**For Lower-End GPUs or CPU:**
```bash
# Reduced settings for limited hardware
python train.py \
  --games-per-iteration 25 \
  --mcts-simulations 400 \
  --batch-size 16 \
  --training-epochs 5
```

## 📊 Expected Performance

### Training Speed (Modern GPU)
- **Self-play**: ~5-15 games/second (depending on hardware)
- **Neural Network**: ~100-500 samples/second inference
- **MCTS**: ~200-1000 simulations/second
- **Complete iteration**: ~20-60 seconds (50 games)

### Training Timeline
- **Beginner level**: 1-2 hours (20-30 iterations)
- **Intermediate level**: 4-6 hours (60-100 iterations)
- **Advanced level**: 8-12 hours (150+ iterations)

### AI Strength Progression
- **Iteration 10**: Beats random play
- **Iteration 30**: Competent tactical play
- **Iteration 60**: Strong strategic understanding
- **Iteration 100+**: Near-optimal play

## 📁 Project Structure

```
Connect4-PyTorch/
├── src/
│   ├── game.py           # Game engine and board representation
│   ├── neural_network.py # AlphaZero neural network
│   ├── mcts.py          # Monte Carlo Tree Search
│   └── self_play.py     # Self-play training data generation
├── models/              # Saved model checkpoints
├── logs/                # Training logs and plots
├── train.py            # Main training script
├── play.py             # Human vs AI interface
└── requirements.txt    # Python dependencies
```

## 🎮 Playing Against the AI

### Game Interface
- Interactive command-line interface
- Real-time move validation
- AI move analysis (optional)
- Game statistics tracking

### AI Difficulty Levels
- **Easy**: 200-400 MCTS simulations
- **Medium**: 800 MCTS simulations (default)
- **Hard**: 1600+ MCTS simulations
- **Expert**: 3200+ MCTS simulations

## 📈 Training Monitoring

### Real-time Monitoring
- Iteration summaries with statistics
- Win rate tracking (should converge to ~50/50)
- Training loss visualization
- Game length trends

### Output Files
- `logs/training_log.json`: Detailed training history
- `logs/training_progress.png`: Training plots
- `logs/training_report.md`: Comprehensive report
- `models/checkpoint_iter_XXXX.pth`: Model checkpoints

### Early Stopping
Training automatically suggests early stopping when:
- Win rates stabilize (variance < 0.001)
- Loss convergence plateaus
- Model performance peaks

## 🔧 Troubleshooting

### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

### Memory Issues
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--mcts-simulations` (try 400)
- Monitor GPU memory: `nvidia-smi`

### Performance Issues
- Ensure GPU is being used: Check for CUDA messages
- Monitor GPU utilization: `nvidia-smi -l 1`
- Try different batch sizes for optimal throughput

### Training Issues
- **Loss not decreasing**: Reduce learning rate
- **Unstable training**: Reduce batch size
- **Memory overflow**: Reduce model size or batch size

## 🚀 Advanced Usage

### Resume Training
```bash
python train.py --resume models/checkpoint_iter_0050.pth
```

### Custom Network Architecture
Edit `src/neural_network.py`:
- Change `num_channels` for wider networks
- Adjust `num_residual_blocks` for deeper networks
- Modify learning rate schedule

### Hyperparameter Tuning
- MCTS simulations vs speed tradeoff
- Temperature scheduling
- Learning rate decay
- Batch size optimization

### Data Analysis
```python
import json
with open('logs/training_log.json') as f:
    data = json.load(f)
# Analyze training metrics
```

## 🧠 How It Works

### AlphaZero Algorithm
1. **Neural Network**: A CNN with residual blocks predicts move probabilities and position values
2. **Monte Carlo Tree Search**: Uses the neural network to guide tree exploration
3. **Self-Play**: The AI plays against itself to generate training data
4. **Training Loop**: The neural network learns from self-play games and improves

### Key Components
- **Dual-Head Network**: Separate outputs for policy (move probabilities) and value (position evaluation)
- **Residual Architecture**: Deep network with skip connections for better gradient flow
- **UCB Selection**: Balances exploration and exploitation in MCTS
- **Temperature Scheduling**: Controls randomness during training vs evaluation

## 🏆 Training Tips

### For Best Results
1. **Start with short training runs** to verify setup
2. **Monitor win rates** - should approach 50/50
3. **Use checkpoints** to prevent data loss
4. **Plot training curves** to detect issues
5. **Gradually increase complexity** as training progresses

### Hyperparameter Guidelines
- **High MCTS simulations** = stronger play, slower training
- **High batch size** = more stable training, more memory
- **High learning rate** = faster convergence, risk instability
- **More epochs** = better learning, risk overfitting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepMind**: For the original AlphaZero algorithm
- **PyTorch Team**: For the excellent deep learning framework
- **Connect 4 Community**: For the classic game that inspired this project

## 📚 References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero Paper)](https://arxiv.org/abs/1712.01815)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Connect 4 Game Theory](https://en.wikipedia.org/wiki/Connect_Four)
- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

---

**Ready to train your own Connect 4 AlphaZero AI! 🎮🤖**

*Star ⭐ this repository if you found it helpful!*