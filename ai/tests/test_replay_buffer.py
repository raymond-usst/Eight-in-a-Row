import pytest
import numpy as np
import os
import shutil
from ai.replay_buffer import ReplayBuffer, GameHistory
from ai.muzero_config import MuZeroConfig

@pytest.fixture
def config():
    cfg = MuZeroConfig()
    cfg.board_size = 10
    cfg.local_view_size = 7
    return cfg

@pytest.fixture
def temp_buffer_dir(tmp_path):
    d = tmp_path / "replay_buffer"
    d.mkdir()
    return str(d)

def create_dummy_game(config, length=10) -> GameHistory:
    history = GameHistory()
    history.actions = list(range(length))
    history.centers = [(5, 5) for _ in range(length)]
    history.policy_targets = [np.ones(config.policy_size)/config.policy_size for _ in range(length)]
    history.root_values = [np.array([0.1, 0.2, 0.3]) for _ in range(length)]
    history.rewards = [0.0] * (length - 1) + [1.0]
    history.player_ids = [(i % 3) + 1 for i in range(length)]
    return history

def test_chunking_and_lru(config, temp_buffer_dir):
    # Setup buffer with very small chunks to force evictions
    buffer = ReplayBuffer(
        max_size=50,
        max_memory_gb=0.01, # very low memory
        priority_alpha=1.0,
        min_games=1,
        chunk_dir=temp_buffer_dir
    )
    
    # 1. Override chunks size internally to trigger saving faster for the test
    buffer.chunk_size = 5 
    
        # Save 15 games
    for i in range(15):
        buffer.save_game(create_dummy_game(config))
    
    assert buffer.num_games() == 15
    
    # It should have saved multiple chunks to disk
    chunk_files = [f for f in os.listdir(temp_buffer_dir) if f.startswith("chunk_") and f.endswith(".pkl")]
    assert len(chunk_files) >= 3

    # Force an LRU load by sampling a specific game index out of cache bounds
    # (By default it keeps active_chunk and maybe a few others, but if memory is low, it evicts)
    # We'll just ask it to load a chunk via sample_batch which randomly picks
    try:
        batch = buffer.sample_batch(
            batch_size=8, 
            num_unroll_steps=5, 
            td_steps=5, 
            discount=0.99, 
            action_size=config.policy_size,
            view_size=config.local_view_size
        )
        assert len(batch) == 14 # 'observations', 'next_observations', 'actions', 'target_values', etc.
        # Check shapes 
        # observations: (B, C, H, W)
        assert batch['observations'].shape == (8, config.observation_channels, config.local_view_size, config.local_view_size)
    except Exception as e:
        pytest.fail(f"sample_batch failed: {e}")

def test_batching_sequence(config, temp_buffer_dir):
    buffer = ReplayBuffer(
        max_size=10,
        chunk_dir=temp_buffer_dir
    )
    
    game = create_dummy_game(config, length=8)
    buffer.save_game(game)
    
    batch_size = 4
    unrolls = 3
    batch = buffer.sample_batch(
        batch_size=batch_size, 
        num_unroll_steps=unrolls, 
        td_steps=5, 
        discount=0.99, 
        action_size=config.policy_size,
        view_size=config.local_view_size
    )
    
    actions = batch['actions']
    targets_value = batch['target_values']
    targets_reward = batch['target_rewards']
    targets_policy = batch['target_policies']
    
    # Check sequences
    assert actions.shape == (batch_size, unrolls)
    assert targets_value.shape == (batch_size, unrolls + 1, 3) # Value exists for T=0 and T=unrolls
    assert targets_reward.shape == (batch_size, unrolls)
    assert targets_policy.shape == (batch_size, unrolls + 1, config.policy_size)
