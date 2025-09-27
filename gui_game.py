#!/usr/bin/env python3
"""
Connect 4 GUI Game
Beautiful graphical interface to play against AlphaZero AI
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import threading
import time
import os
from typing import Optional

from src.neural_network import AlphaZeroNetwork
from src.game import GameState, Player
from src.mcts import MCTS


class Connect4GUI:
    """Beautiful Connect 4 GUI with AI integration"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Connect 4 - AlphaZero AI")
        self.root.geometry("800x700")
        self.root.configure(bg='#2c3e50')

        # Game state
        self.game = GameState()
        self.ai_player = None
        self.human_color = Player.RED
        self.ai_color = Player.YELLOW
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ai_thinking = False

        # Colors
        self.colors = {
            'board': '#34495e',
            'empty': '#ecf0f1',
            'red': '#e74c3c',
            'yellow': '#f1c40f',
            'highlight': '#3498db',
            'text': '#2c3e50',
            'bg': '#2c3e50',
            'button': '#3498db',
            'button_hover': '#2980b9'
        }

        # Board dimensions
        self.rows = 6
        self.cols = 7
        self.cell_size = 70
        self.margin = 10

        self.setup_ui()
        self.new_game()

    def setup_ui(self):
        """Set up the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg=self.colors['bg'])
        title_frame.pack(pady=10)

        title_label = tk.Label(
            title_frame,
            text="🎮 Connect 4 - AlphaZero AI",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg=self.colors['bg']
        )
        title_label.pack()

        # Control panel
        control_frame = tk.Frame(self.root, bg=self.colors['bg'])
        control_frame.pack(pady=10)

        # Game mode selection
        mode_frame = tk.Frame(control_frame, bg=self.colors['bg'])
        mode_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(mode_frame, text="Game Mode:", font=('Arial', 12, 'bold'),
                fg='white', bg=self.colors['bg']).pack()

        self.mode_var = tk.StringVar(value="Human vs Human")
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["Human vs Human", "Human vs AI"],
            state="readonly",
            width=15
        )
        mode_combo.pack(pady=5)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)

        # AI controls
        ai_frame = tk.Frame(control_frame, bg=self.colors['bg'])
        ai_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(ai_frame, text="AI Difficulty:", font=('Arial', 12, 'bold'),
                fg='white', bg=self.colors['bg']).pack()

        self.difficulty_var = tk.StringVar(value="Medium")
        difficulty_combo = ttk.Combobox(
            ai_frame,
            textvariable=self.difficulty_var,
            values=["Easy", "Medium", "Hard", "Expert"],
            state="readonly",
            width=10
        )
        difficulty_combo.pack(pady=5)

        # Buttons
        button_frame = tk.Frame(control_frame, bg=self.colors['bg'])
        button_frame.pack(side=tk.LEFT, padx=10)

        self.load_model_btn = tk.Button(
            button_frame,
            text="📁 Load AI Model",
            command=self.load_ai_model,
            bg=self.colors['button'],
            fg='white',
            font=('Arial', 11, 'bold'),
            relief=tk.FLAT,
            padx=10
        )
        self.load_model_btn.pack(pady=2)

        self.new_game_btn = tk.Button(
            button_frame,
            text="🎯 New Game",
            command=self.new_game,
            bg=self.colors['button'],
            fg='white',
            font=('Arial', 11, 'bold'),
            relief=tk.FLAT,
            padx=10
        )
        self.new_game_btn.pack(pady=2)

        # Status frame
        status_frame = tk.Frame(self.root, bg=self.colors['bg'])
        status_frame.pack(pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Red player's turn",
            font=('Arial', 16, 'bold'),
            fg='white',
            bg=self.colors['bg']
        )
        self.status_label.pack()

        self.ai_status_label = tk.Label(
            status_frame,
            text="",
            font=('Arial', 12),
            fg='#3498db',
            bg=self.colors['bg']
        )
        self.ai_status_label.pack()

        # Game board
        self.setup_board()

        # Statistics
        stats_frame = tk.Frame(self.root, bg=self.colors['bg'])
        stats_frame.pack(pady=10)

        self.stats_label = tk.Label(
            stats_frame,
            text="Games: 0 | Human: 0 | AI: 0 | Draws: 0",
            font=('Arial', 12),
            fg='white',
            bg=self.colors['bg']
        )
        self.stats_label.pack()

        # Game statistics
        self.game_stats = {"total": 0, "human": 0, "ai": 0, "draws": 0}

    def setup_board(self):
        """Set up the game board canvas"""
        board_frame = tk.Frame(self.root, bg=self.colors['bg'])
        board_frame.pack(pady=20)

        canvas_width = self.cols * self.cell_size + 2 * self.margin
        canvas_height = self.rows * self.cell_size + 2 * self.margin

        self.canvas = tk.Canvas(
            board_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.colors['board'],
            relief=tk.RAISED,
            bd=3
        )
        self.canvas.pack()

        # Bind click events
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Motion>', self.on_canvas_hover)

        self.draw_board()

    def draw_board(self):
        """Draw the game board"""
        self.canvas.delete("all")

        # Draw board background
        self.canvas.create_rectangle(
            0, 0,
            self.cols * self.cell_size + 2 * self.margin,
            self.rows * self.cell_size + 2 * self.margin,
            fill=self.colors['board'],
            outline=self.colors['board']
        )

        # Draw grid and pieces
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.cell_size + self.margin + self.cell_size // 2
                y = row * self.cell_size + self.margin + self.cell_size // 2

                piece = self.game.board.board[row, col]

                if piece == Player.EMPTY.value:
                    color = self.colors['empty']
                elif piece == Player.RED.value:
                    color = self.colors['red']
                else:  # YELLOW
                    color = self.colors['yellow']

                # Draw piece with shadow effect
                shadow_offset = 2
                self.canvas.create_oval(
                    x - self.cell_size//2 + 5 + shadow_offset,
                    y - self.cell_size//2 + 5 + shadow_offset,
                    x + self.cell_size//2 - 5 + shadow_offset,
                    y + self.cell_size//2 - 5 + shadow_offset,
                    fill='#1a1a1a',
                    outline='#1a1a1a'
                )

                self.canvas.create_oval(
                    x - self.cell_size//2 + 5,
                    y - self.cell_size//2 + 5,
                    x + self.cell_size//2 - 5,
                    y + self.cell_size//2 - 5,
                    fill=color,
                    outline='#2c3e50',
                    width=2
                )

        # Draw column indicators
        for col in range(self.cols):
            x = col * self.cell_size + self.margin + self.cell_size // 2
            self.canvas.create_text(
                x, 15,
                text=str(col),
                fill='white',
                font=('Arial', 12, 'bold')
            )

    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if self.game.game_over or self.ai_thinking:
            return

        # Calculate column
        col = (event.x - self.margin) // self.cell_size
        if 0 <= col < self.cols:
            self.make_human_move(col)

    def on_canvas_hover(self, event):
        """Handle canvas hover for preview"""
        if self.game.game_over or self.ai_thinking:
            return

        col = (event.x - self.margin) // self.cell_size
        if 0 <= col < self.cols and self.game.board.is_valid_move(col):
            # Highlight column
            self.draw_board()  # Redraw base board

            # Draw preview piece
            for row in range(self.rows - 1, -1, -1):
                if self.game.board.board[row, col] == Player.EMPTY.value:
                    x = col * self.cell_size + self.margin + self.cell_size // 2
                    y = row * self.cell_size + self.margin + self.cell_size // 2

                    color = self.colors['red'] if self.game.current_player == Player.RED else self.colors['yellow']

                    self.canvas.create_oval(
                        x - self.cell_size//2 + 5,
                        y - self.cell_size//2 + 5,
                        x + self.cell_size//2 - 5,
                        y + self.cell_size//2 - 5,
                        fill=color,
                        outline=self.colors['highlight'],
                        width=3,
                        stipple='gray50'
                    )
                    break

    def make_human_move(self, col):
        """Make a human move"""
        if self.mode_var.get() == "Human vs AI" and self.game.current_player == self.ai_color:
            return  # Not human's turn

        if self.game.make_move(col):
            self.draw_board()
            self.update_status()

            if not self.game.game_over and self.mode_var.get() == "Human vs AI" and self.game.current_player == self.ai_color:
                self.make_ai_move()

    def make_ai_move(self):
        """Make an AI move in a separate thread"""
        if not self.ai_player:
            messagebox.showwarning("No AI", "Please load an AI model first!")
            return

        self.ai_thinking = True
        self.ai_status_label.config(text="🤖 AI is thinking...")
        self.root.update()

        # Run AI move in separate thread to prevent GUI freezing
        threading.Thread(target=self._ai_move_thread, daemon=True).start()

    def _ai_move_thread(self):
        """AI move calculation in separate thread"""
        try:
            # Get AI difficulty settings
            difficulty_sims = {
                "Easy": 100,
                "Medium": 400,
                "Hard": 800,
                "Expert": 2500
            }

            simulations = difficulty_sims.get(self.difficulty_var.get(), 400)
            self.ai_player.num_simulations = simulations

            # Get AI move
            action = self.ai_player.get_best_action(self.game, temperature=0.1)

            # Update GUI in main thread
            self.root.after(0, self._complete_ai_move, action)

        except Exception as e:
            self.root.after(0, self._ai_move_error, str(e))

    def _complete_ai_move(self, action):
        """Complete AI move in main thread"""
        self.game.make_move(action)
        self.draw_board()
        self.update_status()
        self.ai_thinking = False
        self.ai_status_label.config(text="")

    def _ai_move_error(self, error_msg):
        """Handle AI move error"""
        self.ai_thinking = False
        self.ai_status_label.config(text="")
        messagebox.showerror("AI Error", f"AI move failed: {error_msg}")

    def update_status(self):
        """Update game status"""
        if self.game.game_over:
            if self.game.winner:
                winner_name = "Red" if self.game.winner == Player.RED else "Yellow"
                if self.mode_var.get() == "Human vs AI":
                    if self.game.winner == self.human_color:
                        winner_name = "🎉 Human"
                        self.game_stats["human"] += 1
                    else:
                        winner_name = "🤖 AI"
                        self.game_stats["ai"] += 1
                else:
                    if self.game.winner == Player.RED:
                        self.game_stats["human"] += 1  # Count as human for stats

                self.status_label.config(text=f"{winner_name} wins!", fg='#27ae60')
            else:
                self.status_label.config(text="It's a draw!", fg='#f39c12')
                self.game_stats["draws"] += 1

            self.game_stats["total"] += 1
            self.update_stats_display()
        else:
            player_name = "Red" if self.game.current_player == Player.RED else "Yellow"
            if self.mode_var.get() == "Human vs AI":
                if self.game.current_player == self.human_color:
                    player_name = "Your"
                else:
                    player_name = "AI's"

            self.status_label.config(text=f"{player_name} turn", fg='white')

    def update_stats_display(self):
        """Update statistics display"""
        stats = self.game_stats
        self.stats_label.config(
            text=f"Games: {stats['total']} | Human: {stats['human']} | AI: {stats['ai']} | Draws: {stats['draws']}"
        )

    def new_game(self):
        """Start a new game"""
        self.game = GameState()
        self.ai_thinking = False
        self.draw_board()
        self.update_status()
        self.ai_status_label.config(text="")

        # If AI should go first
        if (self.mode_var.get() == "Human vs AI" and
            self.ai_player and
            self.game.current_player == self.ai_color):
            self.root.after(1000, self.make_ai_move)  # Delay for better UX

    def on_mode_change(self, event=None):
        """Handle game mode change"""
        if self.mode_var.get() == "Human vs AI" and not self.ai_player:
            messagebox.showinfo("Load AI Model", "Please load an AI model to play against AI!")
            self.mode_var.set("Human vs Human")
        else:
            self.new_game()

    def load_ai_model(self):
        """Load AI model from file"""
        file_path = filedialog.askopenfilename(
            title="Select AI Model",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="models/"
        )

        if file_path:
            try:
                # Load model
                network = AlphaZeroNetwork().to(self.device)

                try:
                    checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
                except Exception:
                    checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)

                if 'network_state_dict' in checkpoint:
                    network.load_state_dict(checkpoint['network_state_dict'])
                    iteration = checkpoint.get('iteration', 'unknown')
                else:
                    network.load_state_dict(checkpoint)
                    iteration = 'unknown'

                network.eval()

                # Create MCTS player
                self.ai_player = MCTS(network, self.device, num_simulations=400)

                # Update UI
                model_name = os.path.basename(file_path)
                self.ai_status_label.config(
                    text=f"✅ Loaded: {model_name} (iter: {iteration})",
                    fg='#27ae60'
                )

                messagebox.showinfo("Success", f"AI model loaded successfully!\nIteration: {iteration}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load AI model:\n{str(e)}")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main function"""
    print("🎮 Starting Connect 4 GUI...")

    app = Connect4GUI()
    app.run()


if __name__ == "__main__":
    main()