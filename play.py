#!/usr/bin/env python3
"""
Play Connect 4 against trained AlphaZero AI
"""

import torch
import argparse
import os
from typing import Optional

from src.neural_network import AlphaZeroNetwork
from src.game import GameState, Player
from src.mcts import MCTS


class HumanPlayer:
    """Human player interface"""

    def __init__(self, color: Player):
        self.color = color

    def get_move(self, game_state: GameState) -> int:
        """Get move from human player"""
        valid_moves = game_state.board.get_valid_moves()

        while True:
            try:
                print(f"\nValid moves: {valid_moves}")
                move = int(input(f"{self.color.name} player, enter column (0-6): "))

                if move in valid_moves:
                    return move
                else:
                    print(f"Invalid move! Choose from {valid_moves}")

            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid number or Ctrl+C to quit")


class AIPlayer:
    """AI player using trained AlphaZero network"""

    def __init__(self, network: AlphaZeroNetwork, device: str, color: Player,
                 mcts_simulations: int = 800, thinking_time: bool = True):
        self.network = network
        self.device = device
        self.color = color
        self.mcts = MCTS(network, device, num_simulations=mcts_simulations)
        self.thinking_time = thinking_time

    def get_move(self, game_state: GameState) -> int:
        """Get move from AI player"""
        if self.thinking_time:
            print(f"\n{self.color.name} AI is thinking...")

        # Get best move using MCTS
        action = self.mcts.get_best_action(game_state, temperature=0.1)

        if self.thinking_time:
            # Show AI's analysis
            action_probs, root = self.mcts.search(game_state, temperature=0.0)
            valid_moves = game_state.board.get_valid_moves()

            print(f"AI analysis:")
            for col in valid_moves:
                prob = action_probs[col]
                visits = root.children[col].visit_count if col in root.children else 0
                print(f"  Column {col}: {prob:.3f} ({visits} visits)")

        return action


def play_game(player1, player2, verbose: bool = True) -> GameState:
    """Play a complete game between two players"""
    game = GameState()
    players = {Player.RED: player1, Player.YELLOW: player2}

    if verbose:
        print("\n" + "="*50)
        print("CONNECT 4 - Human vs AI")
        print("="*50)
        print(game.board)

    while not game.game_over:
        current_player = players[game.current_player]

        try:
            move = current_player.get_move(game)
            success = game.make_move(move)

            if not success:
                print(f"Invalid move by {game.current_player.name}!")
                continue

            if verbose:
                print(f"\n{current_player.color.name} plays column {move}")
                print(game.board)

                if game.game_over:
                    if game.winner:
                        print(f"\n🎉 {game.winner.name} wins!")
                    else:
                        print(f"\n🤝 It's a draw!")

        except KeyboardInterrupt:
            print(f"\nGame interrupted by user")
            break

    return game


def load_model(model_path: str, device: str) -> AlphaZeroNetwork:
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    network = AlphaZeroNetwork().to(device)

    # Load checkpoint with PyTorch 2.6 compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older format or complex objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'network_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['network_state_dict'])
        print(f"Loaded model from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        # Direct state dict
        network.load_state_dict(checkpoint)

    network.eval()
    return network


def main():
    parser = argparse.ArgumentParser(description='Play Connect 4 against AlphaZero AI')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--human-color', type=str, choices=['red', 'yellow'], default='red',
                       help='Human player color (default: red)')
    parser.add_argument('--ai-simulations', type=int, default=800,
                       help='MCTS simulations for AI (default: 800)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--ai-analysis', action='store_true',
                       help='Show AI move analysis')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"🎮 Connect 4 vs AlphaZero")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"AI simulations: {args.ai_simulations}")

    try:
        # Load trained model
        print(f"Loading model...")
        network = load_model(args.model, device)
        print(f"Model loaded successfully!")

        # Set up players
        human_color = Player.RED if args.human_color == 'red' else Player.YELLOW
        ai_color = Player.YELLOW if human_color == Player.RED else Player.RED

        human_player = HumanPlayer(human_color)
        ai_player = AIPlayer(network, device, ai_color, args.ai_simulations, args.ai_analysis)

        print(f"\nGame setup:")
        print(f"Human: {human_color.name}")
        print(f"AI: {ai_color.name}")

        # Play games
        games_played = 0
        human_wins = 0
        ai_wins = 0
        draws = 0

        while True:
            # Determine player order
            if human_color == Player.RED:
                result = play_game(human_player, ai_player)
            else:
                result = play_game(ai_player, human_player)

            games_played += 1

            # Update statistics
            if result.winner == human_color:
                human_wins += 1
            elif result.winner == ai_color:
                ai_wins += 1
            else:
                draws += 1

            print(f"\nGame {games_played} completed!")
            print(f"Score - Human: {human_wins}, AI: {ai_wins}, Draws: {draws}")

            # Ask to play again
            while True:
                try:
                    play_again = input("\nPlay again? (y/n): ").lower().strip()
                    if play_again in ['y', 'yes']:
                        break
                    elif play_again in ['n', 'no']:
                        print(f"\nFinal Score:")
                        print(f"Human: {human_wins}")
                        print(f"AI: {ai_wins}")
                        print(f"Draws: {draws}")
                        print(f"Thanks for playing!")
                        return
                    else:
                        print("Please enter 'y' or 'n'")
                except KeyboardInterrupt:
                    print(f"\nGoodbye!")
                    return

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure you've trained a model first using train.py")
    except KeyboardInterrupt:
        print(f"\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()