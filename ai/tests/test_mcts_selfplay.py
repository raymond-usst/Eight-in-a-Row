import pytest
import torch
import numpy as np
from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.mcts import gumbel_muzero_search
from ai.self_play import SessionContext, play_game

@pytest.fixture
def config():
    cfg = MuZeroConfig()
    cfg.board_size = 10
    cfg.local_view_size = 7
    cfg.policy_size = 49
    # Use very small simulation counts for tests
    cfg.num_simulations = 4
    cfg.num_simulations_play = 4
    cfg.batch_size = 2
    return cfg

@pytest.fixture
def network(config):
    # Fix the device to CPU for testing
    net = MuZeroNetwork(config)
    net.eval()
    return net

def test_session_context():
    ctx = SessionContext(scores={1: 3, 2: 1, 3: 0}, game_idx=2, session_length=5)
    
    # max possible score = 5 * 5 = 25
    # For player 1: 
    # my score = 3/25 = 0.12
    # opp1 (P2) score = 1/25 = 0.04
    # opp2 (P3) score = 0/25 = 0.0
    # games_remaining = (5 - 2 - 1) / 5 = 2/5 = 0.4
    
    vec = ctx.make_context_vector(current_pid=1)
    
    assert len(vec) == 4
    assert np.isclose(vec[0], 0.12)
    assert np.isclose(vec[1], 0.04)
    assert np.isclose(vec[2], 0.0)
    assert np.isclose(vec[3], 0.4)

def test_gumbel_muzero_search(config, network):
    # Dummy inputs without batch dimension
    obs = np.random.randn(config.observation_channels, config.local_view_size, config.local_view_size).astype(np.float32)
    legal_mask = np.ones(config.policy_size, dtype=np.float32)
    
    action_probs, root_value, root = gumbel_muzero_search(
        network, obs, legal_mask, config,
        add_noise=False,
        session_context_vec=None,
        device=torch.device('cpu')
    )
    
    assert action_probs.shape == (config.policy_size,)
    assert np.isclose(np.sum(action_probs), 1.0)
    
    assert isinstance(root_value, np.ndarray)
    assert root_value.shape == (3,)
    
    assert root.visit_count > 0

def test_play_game(config, network):
    # Play a fast tiny game using MCTS
    # Override sizes to force a really quick game
    history = play_game(
        network=network,
        config=config,
        temperature=1.0,
        game_index=0,
        iteration=0,
        board_size_override=5,
        win_length_override=3
    )
    
    # We shouldn't crash, and a game history block should be returned
    assert history is not None
    assert len(history.actions) > 0
    assert len(history.rewards) == len(history.actions)
    assert len(history.root_values) == len(history.actions)
    
    # P1, P2, P3 placement rewards should be scalar floats
    for p in [1, 2, 3]:
        assert isinstance(history.placement_rewards[p], float)
