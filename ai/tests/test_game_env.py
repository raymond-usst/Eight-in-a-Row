import pytest
import numpy as np
from ai.game_env import EightInARowEnv

def test_initialization():
    env = EightInARowEnv(board_size=10, win_length=3)
    assert env.BOARD_SIZE == 10
    assert env.WIN_LENGTH == 3
    assert env.board.shape == (10, 10)
    assert np.all(env.board == 0)
    assert env.current_player == 0

def test_legal_moves_and_masks():
    env = EightInARowEnv(board_size=10, win_length=3)
    env.reset()
    
    # Place a piece to restrict moves
    env.board[5, 5] = 1
    
    # Test local bounds extraction
    moves, mask = env.get_legal_moves_and_mask(5, 5, view_size=5)
    
    # View window: r(3 to 7), c(3 to 7) -> 5x5 = 25 cells
    # One cell is occupied, so 24 legal moves
    assert len(moves) == 24
    assert mask.shape == (25,)
    assert np.sum(mask) == 24
    
    # Assert occupied cell is masked out (center = index 12 in 5x5 flat)
    assert mask[12] == 0.0
    
    # Test boundary view (corner)
    moves_corner, mask_corner = env.get_legal_moves_and_mask(0, 0, view_size=5)
    
    # Center 0,0 with view 5 -> Box is nominally [-2..2, -2..2]
    # Legal board is [0..2, 0..2], so 3x3 = 9 cells are on-board
    # All 9 are empty
    assert len(moves_corner) == 9
    assert np.sum(mask_corner) == 9

def test_rotation_planes():
    env = EightInARowEnv(board_size=10, win_length=3)
    env.reset()
    
    # Player 1 plays
    env.step(1, 1) # board[1,1] = 1, current_player becomes 1 (Player 2)
    # Player 2 plays
    env.step(2, 2) # board[2,2] = 2, current_player becomes 2 (Player 3)
    # Player 3 plays
    env.step(3, 3) # board[3,3] = 3, current_player becomes 0 (Player 1)
    
    # Now it's Player 1's turn (current_player = 0, ID = 1)
    planes_p1 = env._get_rotated_planes_cached()
    
    # planes shape should be [4, 10, 10]
    # Channels: [current_player, next_player, prev_player, empty]
    # For P1: [P1, P2, P3, Empty]
    assert planes_p1[0, 1, 1] == 1.0  # P1's piece
    assert planes_p1[1, 2, 2] == 1.0  # P2's piece
    assert planes_p1[2, 3, 3] == 1.0  # P3's piece
    assert planes_p1[3, 0, 0] == 1.0  # Empty cell
    assert planes_p1[3, 1, 1] == 0.0  # Not empty
    
    # Advance to P2
    env.step(4, 4) # P1 plays, now P2's turn
    planes_p2 = env._get_rotated_planes_cached()
    
    # For P2: [P2, P3, P1, Empty]
    assert planes_p2[0, 2, 2] == 1.0  # Current (P2)
    assert planes_p2[1, 3, 3] == 1.0  # Next (P3)
    assert planes_p2[2, 1, 1] == 1.0  # Prev (P1)

def test_win_detection():
    env = EightInARowEnv(board_size=10, win_length=3)
    env.reset()
    
    # P1 horizontal win
    env.step(0, 0) # P1
    env.step(1, 0) # P2
    env.step(2, 0) # P3
    
    env.step(0, 1) # P1
    env.step(1, 1) # P2
    env.step(2, 1) # P3
    
    rew, done = env.step(0, 2) # P1 connects 3 (0,0), (0,1), (0,2)
    
    assert done == True
    assert env.winner == 1
    assert rew == 1.0 # Winner reward
    # Losers get 2nd (-0.2) or 3rd (-1.0)
    assert env.get_reward(2) in [-0.2, -1.0]
    assert env.get_reward(3) in [-0.2, -1.0]
    
def test_dense_rewards():
    env = EightInARowEnv(board_size=10, win_length=5)
    env.reset()
    
    # We need 5 to win, but step() checks for max_chain >= win_length - 1 for +0.05 reward
    env.board[0, 0] = 1
    env.board[0, 1] = 1
    env.board[0, 2] = 1
    
    # P1 plays 0,3 -> forms a chain of 4 (win_length - 1)
    # The step function should return a reward of 0.05
    env.current_player = 0
    rew, done = env.step(0, 3)
    
    assert done == False
    assert rew == 0.05
