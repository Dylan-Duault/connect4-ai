"""
AlphaZero Neural Network Implementation
High-performance PyTorch implementation optimized for GPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
import time


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""

    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero neural network with policy and value heads
    Optimized for Connect 4 (6x7 board)
    """

    def __init__(self, num_channels: int = 256, num_residual_blocks: int = 8, board_size: Tuple[int, int] = (6, 7)):
        super(AlphaZeroNetwork, self).__init__()

        self.board_height, self.board_width = board_size
        self.num_actions = self.board_width  # 7 columns for Connect 4

        # Initial convolution
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.board_height * self.board_width, self.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * self.board_height * self.board_width, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Input: (batch_size, 3, 6, 7)
        Output: (policy_logits, value)
        """
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value.squeeze(-1)

    def predict(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict policy and value for a single board state
        Returns: (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)

            policy_logits, value = self.forward(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)

            return policy_probs.squeeze(0), value.squeeze(0)

    def get_action_probabilities(self, board_tensor: torch.Tensor, valid_moves: List[int],
                               temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities with temperature scaling and masking
        """
        policy_probs, _ = self.predict(board_tensor)
        policy_probs = policy_probs.cpu().numpy()

        # Mask invalid moves
        masked_probs = np.zeros_like(policy_probs)
        masked_probs[valid_moves] = policy_probs[valid_moves]

        if temperature == 0:
            # Deterministic: choose best valid move
            action_probs = np.zeros_like(masked_probs)
            best_action = valid_moves[np.argmax(masked_probs[valid_moves])]
            action_probs[best_action] = 1.0
        else:
            # Apply temperature
            if temperature != 1.0:
                masked_probs = masked_probs ** (1.0 / temperature)

            # Renormalize
            prob_sum = np.sum(masked_probs)
            if prob_sum > 0:
                action_probs = masked_probs / prob_sum
            else:
                # Fallback: uniform over valid moves
                action_probs = np.zeros_like(masked_probs)
                action_probs[valid_moves] = 1.0 / len(valid_moves)

        return action_probs


class AlphaZeroTrainer:
    """Training manager for AlphaZero network"""

    def __init__(self, network: AlphaZeroNetwork, device: str = 'cuda', learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Training history
        self.training_history = []

    def train_on_batch(self, states: torch.Tensor, target_policies: torch.Tensor,
                      target_values: torch.Tensor) -> Dict[str, float]:
        """Train on a single batch"""
        self.network.train()

        # Forward pass
        policy_logits, predicted_values = self.network(states)

        # Calculate losses
        policy_loss = self.policy_loss_fn(policy_logits, target_policies)
        value_loss = self.value_loss_fn(predicted_values, target_values)
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch"""
        total_losses = {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        num_batches = 0

        start_time = time.time()

        for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            batch_losses = self.train_on_batch(states, target_policies, target_values)

            for key in total_losses:
                total_losses[key] += batch_losses[key]
            num_batches += 1

            if verbose and (batch_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                eta = (len(dataloader) - batch_idx - 1) / batches_per_sec
                print(f"    🧠 Batch {batch_idx + 1:2d}/{len(dataloader):2d}: "
                      f"Loss={batch_losses['total_loss']:.4f} "
                      f"(P:{batch_losses['policy_loss']:.3f}, V:{batch_losses['value_loss']:.3f}) "
                      f"[{batches_per_sec:.1f} batch/sec, ETA: {eta:4.1f}s]")

        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        self.scheduler.step()
        self.training_history.append(avg_losses)

        return avg_losses

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate network on validation data"""
        self.network.eval()
        total_losses = {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        num_batches = 0

        with torch.no_grad():
            for states, target_policies, target_values in dataloader:
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                policy_logits, predicted_values = self.network(states)

                policy_loss = self.policy_loss_fn(policy_logits, target_policies)
                value_loss = self.value_loss_fn(predicted_values, target_values)
                total_loss = policy_loss + value_loss

                total_losses['total_loss'] += total_loss.item()
                total_losses['policy_loss'] += policy_loss.item()
                total_losses['value_loss'] += value_loss.item()
                num_batches += 1

        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    def save_checkpoint(self, filepath: str, epoch: int, additional_info: Dict = None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

        print(f"Checkpoint loaded: {filepath}")
        return checkpoint

    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def create_training_data(states: List[np.ndarray], policies: List[np.ndarray],
                        values: List[float], device: str = 'cuda') -> TensorDataset:
    """
    Create PyTorch dataset from training data
    """
    # Convert to tensors
    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    policies_tensor = torch.FloatTensor(np.array(policies)).to(device)
    values_tensor = torch.FloatTensor(values).to(device)

    return TensorDataset(states_tensor, policies_tensor, values_tensor)


if __name__ == "__main__":
    # Test the neural network
    print("Testing AlphaZero Neural Network...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create network
    network = AlphaZeroNetwork()
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")

    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 6, 7).to(device)

    network = network.to(device)
    start_time = time.time()

    with torch.no_grad():
        policy_logits, values = network(dummy_input)

    inference_time = time.time() - start_time

    print(f"Batch inference time: {inference_time:.4f}s ({batch_size/inference_time:.1f} samples/sec)")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Values shape: {values.shape}")

    # Test single prediction
    single_input = dummy_input[0]
    policy_probs, value = network.predict(single_input)
    print(f"Single prediction - Policy: {policy_probs.shape}, Value: {value.item():.4f}")

    print("Neural network test completed successfully!")