"""
Connect 4 Game Engine
Clean, efficient implementation with GPU-friendly data structures
"""

import numpy as np
import torch
from typing import Optional, Tuple, List
from enum import Enum


class Player(Enum):
    EMPTY = 0
    RED = 1
    YELLOW = 2


class GameBoard:
    """Connect 4 game board with efficient operations"""

    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=np.int8)
        self.move_history = []

    def is_valid_move(self, col: int) -> bool:
        """Check if a column move is valid"""
        return 0 <= col < self.cols and self.board[0, col] == Player.EMPTY.value

    def get_valid_moves(self) -> List[int]:
        """Get all valid column moves"""
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def make_move(self, col: int, player: Player) -> Optional[int]:
        """
        Make a move in specified column
        Returns the row where piece was placed, or None if invalid
        """
        if not self.is_valid_move(col):
            return None

        # Drop piece to lowest available row
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == Player.EMPTY.value:
                self.board[row, col] = player.value
                self.move_history.append((row, col, player))
                return row

        return None

    def undo_move(self) -> bool:
        """Undo the last move"""
        if not self.move_history:
            return False

        row, col, player = self.move_history.pop()
        self.board[row, col] = Player.EMPTY.value
        return True

    def check_winner(self) -> Optional[Player]:
        """Check if there's a winner"""

        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if self._check_line(row, col, 0, 1):
                    return Player(self.board[row, col])

        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if self._check_line(row, col, 1, 0):
                    return Player(self.board[row, col])

        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if self._check_line(row, col, 1, 1):
                    return Player(self.board[row, col])

        # Check diagonal (bottom-left to top-right)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if self._check_line(row, col, -1, 1):
                    return Player(self.board[row, col])

        return None

    def _check_line(self, start_row: int, start_col: int,
                   delta_row: int, delta_col: int) -> bool:
        """Check if there are 4 consecutive pieces in a line"""
        first_piece = self.board[start_row, start_col]
        if first_piece == Player.EMPTY.value:
            return False

        for i in range(1, 4):
            row = start_row + i * delta_row
            col = start_col + i * delta_col
            if self.board[row, col] != first_piece:
                return False

        return True

    def is_full(self) -> bool:
        """Check if board is full"""
        return not any(self.is_valid_move(col) for col in range(self.cols))

    def is_terminal(self) -> bool:
        """Check if game is over"""
        return self.check_winner() is not None or self.is_full()

    def copy(self) -> 'GameBoard':
        """Create a deep copy of the board"""
        new_board = GameBoard(self.rows, self.cols)
        new_board.board = self.board.copy()
        new_board.move_history = self.move_history.copy()
        return new_board

    def to_tensor(self, current_player: Player, device: str = 'cpu') -> torch.Tensor:
        """
        Convert board to neural network input tensor
        Shape: (3, rows, cols)
        Channel 0: Current player pieces
        Channel 1: Opponent pieces
        Channel 2: Current player indicator (all 1s or 0s)
        """
        tensor = torch.zeros(3, self.rows, self.cols, dtype=torch.float32, device=device)

        # Current player pieces
        tensor[0] = torch.tensor(self.board == current_player.value, dtype=torch.float32, device=device)

        # Opponent pieces
        opponent = Player.RED if current_player == Player.YELLOW else Player.YELLOW
        tensor[1] = torch.tensor(self.board == opponent.value, dtype=torch.float32, device=device)

        # Current player indicator
        tensor[2] = torch.full((self.rows, self.cols),
                              1.0 if current_player == Player.RED else 0.0,
                              dtype=torch.float32, device=device)

        return tensor

    def get_symmetrical_board(self) -> 'GameBoard':
        """Get horizontally flipped board (for data augmentation)"""
        flipped = GameBoard(self.rows, self.cols)
        flipped.board = np.fliplr(self.board)
        return flipped

    def __str__(self) -> str:
        """String representation of the board"""
        symbols = {Player.EMPTY.value: '.', Player.RED.value: 'R', Player.YELLOW.value: 'Y'}

        result = "\n  " + " ".join(str(i) for i in range(self.cols)) + "\n"
        for row in range(self.rows):
            result += f"{row} " + " ".join(symbols[self.board[row, col]] for col in range(self.cols)) + "\n"

        return result


class GameState:
    """Complete game state including board and metadata"""

    def __init__(self, board: GameBoard = None, current_player: Player = Player.RED):
        self.board = board if board else GameBoard()
        self.current_player = current_player
        self.winner = None
        self.game_over = False
        self.move_count = 0

    def make_move(self, col: int) -> bool:
        """
        Make a move and update game state
        Returns True if move was successful
        """
        if self.game_over:
            return False

        row = self.board.make_move(col, self.current_player)
        if row is None:
            return False

        self.move_count += 1

        # Check for winner
        self.winner = self.board.check_winner()
        if self.winner or self.board.is_full():
            self.game_over = True
        else:
            # Switch players
            self.current_player = Player.RED if self.current_player == Player.YELLOW else Player.YELLOW

        return True

    def undo_move(self) -> bool:
        """Undo last move and update game state"""
        if self.move_count == 0:
            return False

        if self.board.undo_move():
            self.move_count -= 1
            self.current_player = Player.RED if self.current_player == Player.YELLOW else Player.YELLOW
            self.winner = None
            self.game_over = False
            return True

        return False

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state"""
        new_state = GameState(
            board=self.board.copy(),
            current_player=self.current_player
        )
        new_state.winner = self.winner
        new_state.game_over = self.game_over
        new_state.move_count = self.move_count
        return new_state

    def get_result(self, player: Player) -> float:
        """
        Get game result from perspective of given player
        Returns: 1.0 for win, -1.0 for loss, 0.0 for draw
        """
        if not self.game_over:
            return 0.0

        if self.winner is None:  # Draw
            return 0.0
        elif self.winner == player:
            return 1.0
        else:
            return -1.0

    def __str__(self) -> str:
        status = f"Player: {self.current_player.name}, Move: {self.move_count}"
        if self.game_over:
            if self.winner:
                status += f", Winner: {self.winner.name}"
            else:
                status += ", Draw"

        return f"{status}\n{self.board}"


def play_random_game() -> GameState:
    """Play a random game for testing"""
    import random

    game = GameState()

    while not game.game_over:
        valid_moves = game.board.get_valid_moves()
        if not valid_moves:
            break

        move = random.choice(valid_moves)
        game.make_move(move)

    return game


if __name__ == "__main__":
    # Test the game implementation
    print("Testing Connect 4 Game Engine...")

    # Test basic functionality
    game = GameState()
    print("Initial board:")
    print(game)

    # Make some moves
    moves = [3, 3, 4, 4, 2, 2, 5]  # Should result in Red winning
    for i, move in enumerate(moves):
        success = game.make_move(move)
        print(f"\nMove {i+1}: Column {move} ({'Success' if success else 'Failed'})")
        print(game)

        if game.game_over:
            break

    # Test random game
    print("\n" + "="*50)
    print("Playing random game:")
    random_game = play_random_game()
    print(random_game)

    # Test tensor conversion
    print("\n" + "="*50)
    print("Testing tensor conversion:")
    tensor = game.board.to_tensor(Player.RED)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Current player channel (Red pieces):\n{tensor[0]}")