import pytest
from unittest.mock import MagicMock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.arena import run_arena_match
from ai.muzero_config import MuZeroConfig

class MockGameHistory:
    def __init__(self, winner, length):
        self.winner = winner
        self.length = length

    def __len__(self):
        return self.length

@patch("ai.arena.play_game")
def test_arena_eval_perfect_win(mock_play_game):
    # Model B wins all 10 matches -> Win Rate 1.0, Elo Diff should be +400.0 capped
    
    # Mock play_game to return a succession of GameHistory objects where B wins
    # Recall: when i is even, play_a_as_p1 is True, A is P1,P3; B is P2. B winning means winner=2.
    # when i is odd, play_a_as_p1 is False, B is P1,P3; A is P2. B winning means winner=1 or 3.
    
    def side_effect(network, config, temperature, training_step):
        # We can just cheat by inspecting network keys
        # If network[1] == model_b, B is 1,3
        # If network[1] == model_a, B is 2
        is_a_p1 = (network[1] == "mock_a")
        winner = 2 if is_a_p1 else 1 
        return MockGameHistory(winner=winner, length=30)
    
    mock_play_game.side_effect = side_effect
    
    config = MuZeroConfig()
    
    results = run_arena_match("mock_a", "mock_b", config, num_games=10, verbose=False)
    
    assert results["wins_b"] == 10
    assert results["wins_a"] == 0
    assert results["draws"] == 0
    assert results["win_rate_b"] == 1.0
    assert results["elo_diff"] == 400.0

@patch("ai.arena.play_game")
def test_arena_eval_draws_and_losses(mock_play_game):
    # Simulate: 4 wins B, 2 wins A, 4 Draws
    # Total wins: B=4, A=2, Draw=4. Score = 4 + 0.5 * 4 = 6. Win rate = 6 / 10 = 0.6
    # Expected Elo diff: -400 * log10(1/0.6 - 1) = -400 * log10(2/3) ≈ 70.4
    
    counter = [0]
    def side_effect(network, config, temperature, training_step):
        c = counter[0]
        counter[0] += 1
        is_a_p1 = (network[1] == "mock_a")
        
        # We want: 4 B wins, 2 A wins, 4 draws
        if c < 4:
            winner = 2 if is_a_p1 else 1 # B wins
        elif c < 6:
            winner = 1 if is_a_p1 else 2 # A wins
        else:
            winner = None # Draw
            
        return MockGameHistory(winner=winner, length=30)
    
    mock_play_game.side_effect = side_effect
    
    config = MuZeroConfig()
    
    results = run_arena_match("mock_a", "mock_b", config, num_games=10, verbose=False)
    
    assert results["wins_b"] == 4
    assert results["wins_a"] == 2
    assert results["draws"] == 4
    assert results["win_rate_b"] == 0.6
    assert abs(results["elo_diff"] - 70.436) < 0.1
