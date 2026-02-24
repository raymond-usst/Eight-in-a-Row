
import sys
import os
import torch
import numpy as np
import pytest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.game_env import EightInARowEnv
from ai.muzero_network import MuZeroNetwork
from ai.replay_buffer import ReplayBuffer, GameHistory
from ai.self_play import play_game


@pytest.mark.parametrize("stage,expected_size,expected_win", [
    (1, 15, 5),
    (2, 30, 6),
    (3, 50, 7),
    (4, 100, 8),
])
def test_curriculum_stage(stage, expected_size, expected_win):
    print(f"\nTesting Curriculum Stage {stage} ({expected_size}x{expected_size}, win={expected_win})...")
    
    # 1. Config
    config = MuZeroConfig()
    config.board_size = expected_size
    config.win_length = expected_win
    config.num_simulations_start = config.num_simulations_end = 2 # minimal checks
    config.device = 'cpu'

    # 2. Env
    env = EightInARowEnv(board_size=config.board_size, win_length=config.win_length)
    assert env.BOARD_SIZE == expected_size
    assert env.WIN_LENGTH == expected_win
    assert env.board.shape == (expected_size, expected_size)
    print(f"  [✓] Env created with size {env.BOARD_SIZE}")

    # 3. Network
    network = MuZeroNetwork(config).to(config.device)
    # FocusNet forward check
    dummy_input = torch.randn(1, 4, expected_size, expected_size)
    center_logits = network.focus_net(dummy_input)
    assert center_logits.shape == (1, 2)
    print(f"  [✓] FocusNetwork handled input {expected_size}x{expected_size}")

    # 4. Self Play & History
    # We mock the network's predict_center to avoid complexity or just run play_game
    # play_game uses network.predict_center. FocusNet is initialized so it should work (random weights)
    
    game = play_game(network, config, temperature=1.0, game_index=0, iteration=0)
    assert game.board_size == expected_size
    assert game.win_length == expected_win
    assert len(game) > 0
    print(f"  [✓] Game played. History has size {game.board_size}")

    # 5. Replay Buffer
    buffer = ReplayBuffer(max_size=100)
    buffer.save_game(game)
    assert len(buffer.meta) == 1
    
    # Sample batch
    action_size = expected_size * expected_size
    batch = buffer.sample_batch(batch_size=4, num_unroll_steps=5, td_steps=5, discount=0.99, action_size=action_size)
    
    # Check global state shape in batch
    global_states = batch['global_states'] # (B, 4, Size, Size)
    assert global_states.shape == (4, 4, expected_size, expected_size)
    print(f"  [✓] Batch sampled. Global state shape: {global_states.shape}")

    print(f"Stage {stage} PASSED.")

def main():
    stages = [
        (1, 15, 5),
        (2, 30, 6),
        (3, 50, 7),
        (4, 100, 8)
    ]

    for stage, size, win in stages:
        test_curriculum_stage(stage, size, win)

    print("\nAll curriculum stages verified!")

if __name__ == '__main__':
    main()
