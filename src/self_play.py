"""
Self-Play Training System for AlphaZero
Generates high-quality training data through AI vs AI games
"""

import torch
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import pickle
import os

from .game import GameState, Player
from .neural_network import AlphaZeroNetwork, AlphaZeroTrainer, create_training_data
from .mcts import MCTS


@dataclass
class TrainingExample:
    """Single training example from self-play"""
    board_state: np.ndarray    # Board tensor (3, 6, 7)
    current_player: Player     # Current player
    action_probs: np.ndarray   # MCTS action probabilities (7,)
    game_result: float         # Game outcome from current player's perspective


class SelfPlayManager:
    """Manages self-play game generation and training data collection"""

    def __init__(self, network: AlphaZeroNetwork, device: str = 'cuda',
                 mcts_simulations: int = 800, temperature: float = 1.0,
                 max_game_length: int = 42):
        self.network = network
        self.device = device
        self.mcts = MCTS(network, device, num_simulations=mcts_simulations)
        self.temperature = temperature
        self.max_game_length = max_game_length

        # Training data storage
        self.training_examples = deque(maxlen=100000)  # Keep last 100k examples
        self.game_statistics = {
            'total_games': 0,
            'red_wins': 0,
            'yellow_wins': 0,
            'draws': 0,
            'average_game_length': 0.0,
            'total_examples': 0
        }

    def play_game(self, verbose: bool = False) -> Tuple[List[TrainingExample], Dict]:
        """
        Play a single self-play game and return training examples
        """
        game = GameState()
        game_examples = []
        move_count = 0

        if verbose:
            print(f"Starting self-play game...")

        start_time = time.time()

        while not game.game_over and move_count < self.max_game_length:
            # Get current board state
            board_tensor = game.board.to_tensor(game.current_player, 'cpu').numpy()

            # Use temperature for exploration in early moves
            current_temp = self.temperature if move_count < 10 else 0.1

            # Get action probabilities from MCTS
            action_probs, _ = self.mcts.search(game, temperature=current_temp)

            # Store training example (we'll assign outcomes later)
            example = TrainingExample(
                board_state=board_tensor,
                current_player=game.current_player,
                action_probs=action_probs.copy(),
                game_result=0.0  # Will be filled later
            )
            game_examples.append(example)

            # Sample action from probabilities
            if current_temp == 0:
                action = np.argmax(action_probs)
            else:
                # Ensure probabilities sum to 1
                action_probs_norm = action_probs / np.sum(action_probs)
                action = np.random.choice(len(action_probs_norm), p=action_probs_norm)

            # Make move
            success = game.make_move(action)
            if not success:
                print(f"Warning: Invalid move {action} in self-play")
                break

            move_count += 1

            if verbose and move_count % 5 == 0:
                print(f"  Move {move_count}: Player {game.current_player.name} chose column {action}")

        game_time = time.time() - start_time

        # Assign game outcomes to all examples
        for i, example in enumerate(game_examples):
            if game.winner is None:
                # Draw
                example.game_result = 0.0
            elif game.winner == example.current_player:
                # Win for this player
                example.game_result = 1.0
            else:
                # Loss for this player
                example.game_result = -1.0

        # Update statistics
        self.game_statistics['total_games'] += 1
        if game.winner == Player.RED:
            self.game_statistics['red_wins'] += 1
        elif game.winner == Player.YELLOW:
            self.game_statistics['yellow_wins'] += 1
        else:
            self.game_statistics['draws'] += 1

        # Update average game length
        total_length = (self.game_statistics['average_game_length'] *
                       (self.game_statistics['total_games'] - 1) + move_count)
        self.game_statistics['average_game_length'] = total_length / self.game_statistics['total_games']

        game_result = {
            'winner': game.winner,
            'game_length': move_count,
            'game_time': game_time,
            'examples_generated': len(game_examples)
        }

        if verbose:
            winner_str = game.winner.name if game.winner else "Draw"
            print(f"Game completed: {winner_str} in {move_count} moves ({game_time:.2f}s)")

        return game_examples, game_result

    def generate_training_data(self, num_games: int, verbose: bool = True) -> List[TrainingExample]:
        """Generate training data from multiple self-play games"""
        new_examples = []
        start_time = time.time()

        if verbose:
            print(f"Generating training data from {num_games} self-play games...")

        for game_num in range(num_games):
            game_examples, game_result = self.play_game(verbose=False)
            new_examples.extend(game_examples)

            if verbose:
                elapsed = time.time() - start_time
                games_per_sec = (game_num + 1) / elapsed
                eta = (num_games - game_num - 1) / games_per_sec

                winner_str = game_result['winner'].name if game_result['winner'] else "Draw"

                # Show every game for first 10, then every 5th, then every 10th
                show_game = (game_num < 10) or ((game_num + 1) % 5 == 0 and game_num < 50) or ((game_num + 1) % 10 == 0)

                if show_game:
                    print(f"  🎮 Game {game_num + 1:3d}/{num_games}: {winner_str:6s} "
                          f"({game_result['game_length']:2d} moves, {game_result['examples_generated']:2d} examples) "
                          f"[{games_per_sec:.1f} games/sec, ETA: {eta:4.0f}s]")
                elif (game_num + 1) % 25 == 0:
                    # Show milestone updates
                    red_rate = self.game_statistics['red_wins'] / self.game_statistics['total_games'] * 100
                    yellow_rate = self.game_statistics['yellow_wins'] / self.game_statistics['total_games'] * 100
                    draw_rate = self.game_statistics['draws'] / self.game_statistics['total_games'] * 100
                    print(f"  📊 Progress {game_num + 1:3d}/{num_games}: "
                          f"R:{red_rate:4.1f}% Y:{yellow_rate:4.1f}% D:{draw_rate:4.1f}% "
                          f"[{games_per_sec:.1f} games/sec, ETA: {eta:4.0f}s]")

        # Add to training data buffer
        self.training_examples.extend(new_examples)
        self.game_statistics['total_examples'] = len(self.training_examples)

        total_time = time.time() - start_time
        if verbose:
            print(f"Generated {len(new_examples)} training examples in {total_time:.1f}s "
                  f"({len(new_examples)/total_time:.1f} examples/sec)")

        return new_examples

    def get_training_data(self, num_examples: Optional[int] = None) -> List[TrainingExample]:
        """Get training data for neural network training"""
        if num_examples is None or num_examples >= len(self.training_examples):
            return list(self.training_examples)

        # Randomly sample examples
        return random.sample(list(self.training_examples), num_examples)

    def prepare_training_tensors(self, examples: List[TrainingExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert training examples to PyTorch tensors"""
        states = []
        policies = []
        values = []

        for example in examples:
            states.append(example.board_state)
            policies.append(example.action_probs)
            values.append(example.game_result)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(policies)),
            torch.FloatTensor(values)
        )

    def save_training_data(self, filepath: str):
        """Save training data to disk"""
        data = {
            'training_examples': list(self.training_examples),
            'game_statistics': self.game_statistics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Training data saved: {filepath} ({len(self.training_examples)} examples)")

    def load_training_data(self, filepath: str):
        """Load training data from disk"""
        if not os.path.exists(filepath):
            print(f"Training data file not found: {filepath}")
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.training_examples = deque(data['training_examples'], maxlen=self.training_examples.maxlen)
        self.game_statistics = data.get('game_statistics', self.game_statistics)

        print(f"Training data loaded: {filepath} ({len(self.training_examples)} examples)")

    def get_statistics(self) -> Dict:
        """Get current training statistics"""
        stats = self.game_statistics.copy()

        if stats['total_games'] > 0:
            stats['red_win_rate'] = stats['red_wins'] / stats['total_games']
            stats['yellow_win_rate'] = stats['yellow_wins'] / stats['total_games']
            stats['draw_rate'] = stats['draws'] / stats['total_games']
        else:
            stats['red_win_rate'] = 0.0
            stats['yellow_win_rate'] = 0.0
            stats['draw_rate'] = 0.0

        return stats

    def clear_training_data(self):
        """Clear all training data"""
        self.training_examples.clear()
        self.game_statistics = {
            'total_games': 0,
            'red_wins': 0,
            'yellow_wins': 0,
            'draws': 0,
            'average_game_length': 0.0,
            'total_examples': 0
        }


class AlphaZeroTrainingLoop:
    """Complete AlphaZero training loop"""

    def __init__(self, network: AlphaZeroNetwork, device: str = 'cuda',
                 mcts_simulations: int = 800, learning_rate: float = 0.001):
        self.network = network
        self.device = device
        self.trainer = AlphaZeroTrainer(network, device, learning_rate)
        self.self_play = SelfPlayManager(network, device, mcts_simulations)

        # Training configuration
        self.iteration = 0
        self.training_history = []

    def run_iteration(self, num_games: int = 100, training_epochs: int = 10,
                     batch_size: int = 32, verbose: bool = True) -> Dict:
        """Run one training iteration"""
        self.iteration += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING ITERATION {self.iteration}")
            print(f"{'='*60}")

        iteration_start = time.time()

        # 1. Generate self-play data
        if verbose:
            print(f"\n1. Generating self-play data ({num_games} games)...")

        new_examples = self.self_play.generate_training_data(num_games, verbose)

        # 2. Train neural network
        if len(self.self_play.training_examples) >= batch_size:
            if verbose:
                print(f"\n2. Training neural network ({training_epochs} epochs)...")

            # Get training data
            training_examples = self.self_play.get_training_data()
            states, policies, values = self.self_play.prepare_training_tensors(training_examples)

            # Create data loader
            dataset = create_training_data(
                states.numpy(), policies.numpy(), values.numpy(), self.device
            )
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Train for multiple epochs
            epoch_losses = []
            for epoch in range(training_epochs):
                if verbose:
                    print(f"\n  Epoch {epoch + 1}/{training_epochs}:")

                losses = self.trainer.train_epoch(dataloader, verbose)
                epoch_losses.append(losses)

                if verbose:
                    print(f"    Loss: {losses['total_loss']:.4f} "
                          f"(Policy: {losses['policy_loss']:.4f}, "
                          f"Value: {losses['value_loss']:.4f})")

            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
        else:
            if verbose:
                print(f"\n2. Skipping training (need {batch_size} examples, have {len(self.self_play.training_examples)})")
            avg_losses = {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        iteration_time = time.time() - iteration_start

        # 3. Collect statistics
        stats = self.self_play.get_statistics()
        iteration_result = {
            'iteration': self.iteration,
            'games_played': num_games,
            'new_examples': len(new_examples),
            'total_examples': len(self.self_play.training_examples),
            'training_losses': avg_losses,
            'game_statistics': stats,
            'iteration_time': iteration_time
        }

        self.training_history.append(iteration_result)

        if verbose:
            print(f"\n3. Iteration {self.iteration} Summary:")
            print(f"   Games played: {num_games}")
            print(f"   New examples: {len(new_examples)}")
            print(f"   Total examples: {len(self.self_play.training_examples)}")
            print(f"   Red wins: {stats['red_win_rate']:.1%}")
            print(f"   Yellow wins: {stats['yellow_win_rate']:.1%}")
            print(f"   Draws: {stats['draw_rate']:.1%}")
            print(f"   Avg game length: {stats['average_game_length']:.1f}")
            print(f"   Training loss: {avg_losses['total_loss']:.4f}")
            print(f"   Iteration time: {iteration_time:.1f}s")

        return iteration_result

    def save_checkpoint(self, filepath: str):
        """Save complete training checkpoint"""
        checkpoint_data = {
            'iteration': self.iteration,
            'training_history': self.training_history,
            'game_statistics': self.self_play.get_statistics()
        }

        # Save neural network
        self.trainer.save_checkpoint(filepath, self.iteration, checkpoint_data)

        # Save training data
        data_filepath = filepath.replace('.pth', '_data.pkl')
        self.self_play.save_training_data(data_filepath)

    def load_checkpoint(self, filepath: str):
        """Load complete training checkpoint"""
        # Load neural network
        checkpoint = self.trainer.load_checkpoint(filepath)

        self.iteration = checkpoint.get('iteration', 0)
        self.training_history = checkpoint.get('training_history', [])

        # Load training data
        data_filepath = filepath.replace('.pth', '_data.pkl')
        self.self_play.load_training_data(data_filepath)


if __name__ == "__main__":
    # Test self-play system
    print("Testing Self-Play Training System...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network and training system
    network = AlphaZeroNetwork().to(device)
    training_loop = AlphaZeroTrainingLoop(network, device, mcts_simulations=100)

    # Test single self-play game
    print("\nTesting single self-play game...")
    examples, game_result = training_loop.self_play.play_game(verbose=True)

    print(f"Generated {len(examples)} training examples")
    print(f"Game result: {game_result}")

    # Test training iteration
    print("\nTesting training iteration...")
    result = training_loop.run_iteration(num_games=5, training_epochs=2, verbose=True)

    print(f"Iteration completed successfully!")
    print(f"Final statistics: {training_loop.self_play.get_statistics()}")

    print("Self-play test completed successfully!")