#!/usr/bin/env python3
"""
AlphaZero Connect 4 Training Script
High-performance training with GPU acceleration and monitoring
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, List

# Import our modules
from src.neural_network import AlphaZeroNetwork
from src.self_play import AlphaZeroTrainingLoop


class TrainingMonitor:
    """Training progress monitoring and visualization"""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.training_log = []
        self.start_time = time.time()

    def log_iteration(self, iteration_result: Dict):
        """Log iteration results"""
        iteration_result['timestamp'] = datetime.now().isoformat()
        iteration_result['elapsed_time'] = time.time() - self.start_time

        self.training_log.append(iteration_result)

        # Save to JSON
        log_file = os.path.join(self.log_dir, 'training_log.json')
        with open(log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2, default=str)

        # Print summary
        self._print_iteration_summary(iteration_result)

    def _print_iteration_summary(self, result: Dict):
        """Print iteration summary"""
        iteration = result['iteration']
        games = result['games_played']
        examples = result['total_examples']
        time_taken = result['iteration_time']
        elapsed_total = result['elapsed_time']

        stats = result['game_statistics']
        loss = result['training_losses']['total_loss']

        # Calculate rates
        games_per_sec = games / time_taken
        examples_per_sec = result['new_examples'] / time_taken

        print(f"\n📊 ITERATION {iteration} SUMMARY")
        print(f"   🎮 Games: {games} in {time_taken:.1f}s ({games_per_sec:.1f} games/sec)")
        print(f"   📝 Examples: +{result['new_examples']} ({examples_per_sec:.1f}/sec) → {examples:,} total")
        print(f"   🏆 Win rates: Red {stats['red_win_rate']:.1%}, "
              f"Yellow {stats['yellow_win_rate']:.1%}, "
              f"Draw {stats['draw_rate']:.1%}")
        print(f"   📏 Avg game length: {stats['average_game_length']:.1f} moves")
        print(f"   🧠 Training loss: {loss:.4f}")
        print(f"   ⏱️  Total elapsed: {elapsed_total/60:.1f} minutes")
        print(f"   {'='*60}")

    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        if len(self.training_log) < 2:
            print("Not enough data to plot")
            return

        iterations = [r['iteration'] for r in self.training_log]
        losses = [r['training_losses']['total_loss'] for r in self.training_log]
        red_wins = [r['game_statistics']['red_win_rate'] for r in self.training_log]
        yellow_wins = [r['game_statistics']['yellow_win_rate'] for r in self.training_log]
        game_lengths = [r['game_statistics']['average_game_length'] for r in self.training_log]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AlphaZero Training Progress', fontsize=16)

        # Training loss
        ax1.plot(iterations, losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        # Win rates
        ax2.plot(iterations, red_wins, 'r-', linewidth=2, label='Red')
        ax2.plot(iterations, yellow_wins, 'y-', linewidth=2, label='Yellow')
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Expected (50%)')
        ax2.set_title('Win Rates')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Win Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Game length
        ax3.plot(iterations, game_lengths, 'g-', linewidth=2)
        ax3.set_title('Average Game Length')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Moves')
        ax3.grid(True, alpha=0.3)

        # Cumulative examples
        cumulative_examples = [r['total_examples'] for r in self.training_log]
        ax4.plot(iterations, cumulative_examples, 'm-', linewidth=2)
        ax4.set_title('Training Data Size')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Examples')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved: {save_path}")

        plt.show()

    def generate_report(self, save_path: str = None) -> str:
        """Generate training report"""
        if not self.training_log:
            return "No training data available"

        latest = self.training_log[-1]
        total_time = latest['elapsed_time']

        report = f"""
# AlphaZero Connect 4 Training Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Summary

- **Total Iterations:** {latest['iteration']}
- **Total Training Time:** {total_time/3600:.2f} hours
- **Total Games Played:** {latest['game_statistics']['total_games']:,}
- **Total Training Examples:** {latest['total_examples']:,}

## Final Performance

- **Red Win Rate:** {latest['game_statistics']['red_win_rate']:.1%}
- **Yellow Win Rate:** {latest['game_statistics']['yellow_win_rate']:.1%}
- **Draw Rate:** {latest['game_statistics']['draw_rate']:.1%}
- **Average Game Length:** {latest['game_statistics']['average_game_length']:.1f} moves
- **Final Training Loss:** {latest['training_losses']['total_loss']:.4f}

## Training Progress

| Iteration | Games | Examples | Red Win% | Yellow Win% | Draw% | Loss | Time(s) |
|-----------|-------|----------|----------|-------------|-------|------|---------|
"""

        for result in self.training_log[-10:]:  # Last 10 iterations
            report += f"| {result['iteration']} | {result['games_played']} | {result['total_examples']:,} | "
            report += f"{result['game_statistics']['red_win_rate']:.1%} | "
            report += f"{result['game_statistics']['yellow_win_rate']:.1%} | "
            report += f"{result['game_statistics']['draw_rate']:.1%} | "
            report += f"{result['training_losses']['total_loss']:.3f} | "
            report += f"{result['iteration_time']:.1f} |\n"

        report += "\n---\n*Generated by AlphaZero Like Training System*"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Training report saved: {save_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Connect 4 Training')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of training iterations (default: 100)')
    parser.add_argument('--games-per-iteration', type=int, default=50,
                       help='Number of self-play games per iteration (default: 50)')
    parser.add_argument('--mcts-simulations', type=int, default=800,
                       help='MCTS simulations per move (default: 800)')
    parser.add_argument('--training-epochs', type=int, default=10,
                       help='Neural network training epochs per iteration (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                       help='Save checkpoint every N iterations (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"🚀 AlphaZero Connect 4 Training")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games_per_iteration}")
    print(f"MCTS simulations: {args.mcts_simulations}")
    print(f"Training epochs: {args.training_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Initialize network and training
    network = AlphaZeroNetwork().to(device)
    training_loop = AlphaZeroTrainingLoop(
        network, device, args.mcts_simulations, args.learning_rate
    )
    monitor = TrainingMonitor()

    # Resume from checkpoint if specified
    if args.resume:
        print(f"📁 Resuming from checkpoint: {args.resume}")
        training_loop.load_checkpoint(args.resume)

    print(f"\n🎯 Starting training...")
    print(f"Neural network parameters: {sum(p.numel() for p in network.parameters()):,}")

    try:
        # Training statistics tracking
        total_start_time = time.time()

        # Main training loop
        for iteration in range(training_loop.iteration + 1, args.iterations + 1):
            # Calculate and show ETA
            if iteration > training_loop.iteration + 1:
                elapsed = time.time() - total_start_time
                avg_time_per_iter = elapsed / (iteration - training_loop.iteration - 1)
                remaining_iters = args.iterations - iteration + 1
                eta_seconds = avg_time_per_iter * remaining_iters
                eta_hours = eta_seconds / 3600

                print(f"\n⏰ Starting iteration {iteration}/{args.iterations} "
                      f"(ETA: {eta_hours:.1f}h remaining)")

            result = training_loop.run_iteration(
                num_games=args.games_per_iteration,
                training_epochs=args.training_epochs,
                batch_size=args.batch_size,
                verbose=True
            )

            monitor.log_iteration(result)

            # Save checkpoint
            if iteration % args.checkpoint_freq == 0:
                checkpoint_path = f'models/checkpoint_iter_{iteration:04d}.pth'
                training_loop.save_checkpoint(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

                # Show overall progress
                total_elapsed = time.time() - total_start_time
                progress_pct = (iteration - training_loop.iteration) / args.iterations * 100
                print(f"🚀 Overall progress: {progress_pct:.1f}% complete "
                      f"({total_elapsed/3600:.1f}h elapsed)")

            # Early stopping check
            if iteration >= 10:
                recent_results = monitor.training_log[-5:]
                win_rate_variance = np.var([r['game_statistics']['red_win_rate'] for r in recent_results])

                if win_rate_variance < 0.001:  # Very stable win rates
                    print(f"\n🎯 Training appears to be converging (win rate variance: {win_rate_variance:.6f})")
                    print("Consider early stopping or reducing learning rate.")

    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")

    # Final checkpoint and analysis
    final_checkpoint = f'models/final_model.pth'
    training_loop.save_checkpoint(final_checkpoint)
    print(f"💾 Final model saved: {final_checkpoint}")

    # Generate report and plots
    print(f"\n📊 Generating training analysis...")
    plot_path = f'logs/training_progress.png'
    monitor.plot_training_progress(plot_path)

    report_path = f'logs/training_report.md'
    monitor.generate_report(report_path)

    print(f"\n✅ Training completed!")
    print(f"📈 Training plot: {plot_path}")
    print(f"📋 Training report: {report_path}")
    print(f"🎮 Final model: {final_checkpoint}")

    # Final statistics
    stats = training_loop.self_play.get_statistics()
    print(f"\n🏆 Final Statistics:")
    print(f"   Total games: {stats['total_games']:,}")
    print(f"   Training examples: {stats['total_examples']:,}")
    print(f"   Red win rate: {stats['red_win_rate']:.1%}")
    print(f"   Yellow win rate: {stats['yellow_win_rate']:.1%}")
    print(f"   Draw rate: {stats['draw_rate']:.1%}")
    print(f"   Average game length: {stats['average_game_length']:.1f} moves")


if __name__ == "__main__":
    main()