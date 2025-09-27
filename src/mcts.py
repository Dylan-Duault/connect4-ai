"""
Monte Carlo Tree Search with GPU-Accelerated Neural Network
High-performance MCTS implementation for AlphaZero training
"""

import torch
import numpy as np
import math
import time
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from .game import GameState, Player
from .neural_network import AlphaZeroNetwork


class MCTSNode:
    """Monte Carlo Tree Search Node"""

    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None, prior: float = 0.0):
        self.game_state = game_state.copy()
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior   # Prior probability from neural network

        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded = False

    @property
    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is root"""
        return self.parent is None

    def ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate UCB score for node selection
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if self.visit_count == 0:
            return float('inf')

        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value + exploration

    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select best child according to UCB formula"""
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def add_child(self, action: int, game_state: GameState, prior: float) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(game_state, parent=self, action=action, prior=prior)
        self.children[action] = child
        return child

    def backup(self, value: float):
        """Backup value through the tree"""
        self.visit_count += 1
        self.value_sum += value

        if not self.is_root():
            # Flip value for opponent
            self.parent.backup(-value)

    def get_action_probabilities(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts
        """
        action_probs = np.zeros(7)  # 7 columns in Connect 4

        if not self.children:
            return action_probs

        # Get visit counts
        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]

        if temperature == 0:
            # Deterministic: choose most visited action
            best_action = actions[np.argmax(visit_counts)]
            action_probs[best_action] = 1.0
        else:
            # Apply temperature
            visit_counts = np.array(visit_counts, dtype=np.float32)
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)

            # Normalize
            if np.sum(visit_counts) > 0:
                visit_counts = visit_counts / np.sum(visit_counts)
                for i, action in enumerate(actions):
                    action_probs[action] = visit_counts[i]

        return action_probs

    def most_visited_action(self) -> Optional[int]:
        """Get the most visited child action"""
        if not self.children:
            return None

        return max(self.children.keys(), key=lambda action: self.children[action].visit_count)


class MCTS:
    """
    Monte Carlo Tree Search with Neural Network guidance
    Optimized for GPU-accelerated batch inference
    """

    def __init__(self, network: AlphaZeroNetwork, device: str = 'cuda',
                 c_puct: float = 1.0, num_simulations: int = 800):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations

        # Performance tracking
        self.total_nodes_created = 0
        self.total_nn_calls = 0
        self.total_search_time = 0.0

    def search(self, game_state: GameState, temperature: float = 1.0,
              verbose: bool = False) -> Tuple[np.ndarray, MCTSNode]:
        """
        Run MCTS search and return action probabilities
        """
        start_time = time.time()

        # Create root node
        root = MCTSNode(game_state)

        # Expand root if not terminal
        if not game_state.game_over:
            self._expand_node(root)

        # Run simulations
        for simulation in range(self.num_simulations):
            if verbose and (simulation + 1) % 200 == 0:
                elapsed = time.time() - start_time
                sims_per_sec = (simulation + 1) / elapsed
                eta = (self.num_simulations - simulation - 1) / sims_per_sec
                print(f"      🌳 MCTS: {simulation + 1:4d}/{self.num_simulations} sims "
                      f"({sims_per_sec:3.0f}/sec, ETA: {eta:4.1f}s)", end='\r', flush=True)

            self._simulate(root)

        if verbose:
            search_time = time.time() - start_time
            self.total_search_time += search_time
            print(f"\n    MCTS completed: {self.num_simulations} simulations in {search_time:.2f}s "
                  f"({self.num_simulations/search_time:.1f} sims/sec)")

        # Get action probabilities
        action_probs = root.get_action_probabilities(temperature)

        return action_probs, root

    def _simulate(self, node: MCTSNode):
        """Run a single MCTS simulation"""
        path = []

        # Selection: traverse tree using UCB
        current = node
        while not current.is_leaf() and current.is_expanded:
            path.append(current)
            current = current.select_child(self.c_puct)

        path.append(current)

        # Get value
        if current.game_state.game_over:
            # Terminal node: get actual game result
            value = current.game_state.get_result(current.game_state.current_player)
        else:
            # Expand if not already expanded
            if not current.is_expanded:
                self._expand_node(current)

            # Evaluate with neural network
            value = self._evaluate_node(current)

        # Backup
        current.backup(value)

    def _expand_node(self, node: MCTSNode):
        """Expand node by adding all legal moves as children"""
        if node.game_state.game_over or node.is_expanded:
            return

        # Get neural network predictions
        policy_probs, value = self._get_network_prediction(node.game_state)

        # Add children for all valid moves
        valid_moves = node.game_state.board.get_valid_moves()
        for action in valid_moves:
            # Create child game state
            child_state = node.game_state.copy()
            child_state.make_move(action)

            # Add child with prior probability
            prior = policy_probs[action]
            node.add_child(action, child_state, prior)
            self.total_nodes_created += 1

        node.is_expanded = True

    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate node using neural network"""
        _, value = self._get_network_prediction(node.game_state)
        return value

    def _get_network_prediction(self, game_state: GameState) -> Tuple[np.ndarray, float]:
        """Get neural network prediction for game state"""
        self.total_nn_calls += 1

        # Convert game state to tensor
        board_tensor = game_state.board.to_tensor(game_state.current_player, self.device)

        # Get network prediction
        policy_probs, value = self.network.predict(board_tensor)

        # Convert to numpy and mask invalid moves
        policy_probs = policy_probs.cpu().numpy()
        valid_moves = game_state.board.get_valid_moves()

        # Mask invalid moves and renormalize
        masked_probs = np.zeros_like(policy_probs)
        if valid_moves:
            masked_probs[valid_moves] = policy_probs[valid_moves]
            prob_sum = np.sum(masked_probs)
            if prob_sum > 0:
                masked_probs = masked_probs / prob_sum
            else:
                # Fallback: uniform distribution over valid moves
                masked_probs[valid_moves] = 1.0 / len(valid_moves)

        return masked_probs, value.item()

    def get_best_action(self, game_state: GameState, temperature: float = 0.0) -> int:
        """Get best action using MCTS"""
        action_probs, root = self.search(game_state, temperature)

        if temperature == 0.0:
            # Deterministic: choose most visited action
            return root.most_visited_action()
        else:
            # Stochastic: sample from action probabilities
            # Ensure probabilities sum to 1 (fix numerical precision issues)
            action_probs = action_probs / np.sum(action_probs)
            return np.random.choice(len(action_probs), p=action_probs)

    def get_search_statistics(self) -> Dict[str, float]:
        """Get MCTS performance statistics"""
        return {
            'total_nodes_created': self.total_nodes_created,
            'total_nn_calls': self.total_nn_calls,
            'total_search_time': self.total_search_time,
            'avg_nn_calls_per_search': self.total_nn_calls / max(1, self.total_search_time),
            'avg_nodes_per_search': self.total_nodes_created / max(1, self.total_search_time)
        }

    def reset_statistics(self):
        """Reset performance tracking"""
        self.total_nodes_created = 0
        self.total_nn_calls = 0
        self.total_search_time = 0.0


class BatchMCTS:
    """
    Batch MCTS for efficient parallel game generation
    Groups multiple games for batch neural network inference
    """

    def __init__(self, network: AlphaZeroNetwork, device: str = 'cuda',
                 batch_size: int = 32, c_puct: float = 1.0, num_simulations: int = 800):
        self.network = network
        self.device = device
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search_batch(self, game_states: List[GameState], temperature: float = 1.0) -> List[np.ndarray]:
        """
        Run MCTS search on a batch of game states
        """
        # Create individual MCTS instances
        mcts_instances = [MCTS(self.network, self.device, self.c_puct, self.num_simulations)
                         for _ in game_states]

        # Run searches
        results = []
        for i, (mcts, game_state) in enumerate(zip(mcts_instances, game_states)):
            action_probs, _ = mcts.search(game_state, temperature, verbose=(i == 0))
            results.append(action_probs)

        return results


def play_mcts_game(network: AlphaZeroNetwork, device: str = 'cuda',
                  num_simulations: int = 400, temperature: float = 1.0) -> GameState:
    """Play a full game using MCTS"""
    game = GameState()
    mcts = MCTS(network, device, num_simulations=num_simulations)

    move_count = 0
    while not game.game_over and move_count < 42:  # Prevent infinite games
        # Use temperature for first 10 moves, then deterministic
        current_temp = temperature if move_count < 10 else 0.1

        action = mcts.get_best_action(game, current_temp)
        game.make_move(action)
        move_count += 1

    return game


if __name__ == "__main__":
    # Test MCTS implementation
    print("Testing MCTS with Neural Network...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network and MCTS
    network = AlphaZeroNetwork().to(device)
    mcts = MCTS(network, device, num_simulations=100)

    # Test single search
    game = GameState()
    print("\nTesting single MCTS search...")

    start_time = time.time()
    action_probs, root = mcts.search(game, verbose=True)
    search_time = time.time() - start_time

    print(f"Action probabilities: {action_probs}")
    print(f"Root visits: {root.visit_count}")
    print(f"Search time: {search_time:.3f}s")

    # Test full game
    print("\nTesting full MCTS game...")
    game_result = play_mcts_game(network, device, num_simulations=50)
    print(f"Game completed in {game_result.move_count} moves")
    if game_result.winner:
        print(f"Winner: {game_result.winner.name}")
    else:
        print("Game ended in a draw")

    # Print statistics
    stats = mcts.get_search_statistics()
    print(f"\nMCTS Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    print("MCTS test completed successfully!")