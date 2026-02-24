import pytest
import torch
from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork

@pytest.fixture
def config():
    cfg = MuZeroConfig()
    cfg.board_size = 10
    cfg.local_view_size = 7
    return cfg

@pytest.fixture
def network(config):
    return MuZeroNetwork(config)

def test_representation(config, network):
    batch_size = 4
    obs = torch.randn(batch_size, config.observation_channels, config.local_view_size, config.local_view_size)
    hidden_state = network.representation(obs)
    
    assert hidden_state.shape == (batch_size, config.hidden_state_dim)
    assert not torch.isnan(hidden_state).any()

def test_dynamics(config, network):
    batch_size = 4
    hidden_state = torch.randn(batch_size, config.hidden_state_dim)
    action = torch.randint(0, config.policy_size, (batch_size,))
    
    action_one_hot = torch.zeros(batch_size, config.policy_size)
    action_one_hot.scatter_(1, action.unsqueeze(1), 1.0)
    
    next_state, reward = network.dynamics(hidden_state, action_one_hot)
    
    assert next_state.shape == (batch_size, config.hidden_state_dim)
    assert reward.shape == (batch_size,)
    
    assert not torch.isnan(next_state).any()
    assert not torch.isnan(reward).any()

def test_prediction(config, network):
    batch_size = 4
    hidden_state = torch.randn(batch_size, config.hidden_state_dim)
    
    policy_logits, value, threat_logits, opp_logits, heatmap_logits = network.prediction(hidden_state)
    
    assert policy_logits.shape == (batch_size, config.policy_size)
    assert value.shape == (batch_size, 3)
    
    assert not torch.isnan(policy_logits).any()
    assert not torch.isnan(value).any()

def test_reconstruction(config, network):
    batch_size = 4
    hidden_state = torch.randn(batch_size, config.hidden_state_dim)
    
    recon = network.reconstruct_state(hidden_state)
    
    assert recon.shape == (batch_size, 8, config.local_view_size, config.local_view_size)
    assert not torch.isnan(recon).any()

def test_unrolled_stability(config, network):
    batch_size = 4
    unroll_steps = 10
    
    obs = torch.randn(batch_size, config.observation_channels, config.local_view_size, config.local_view_size)
    hidden_state = network.representation(obs)
    
    for _ in range(unroll_steps):
        policy_logits, value, threat_logits, opp_logits, heatmap_logits = network.prediction(hidden_state)
        action_idx = torch.argmax(policy_logits, dim=1)
        
        action_one_hot = torch.zeros(batch_size, config.policy_size)
        action_one_hot.scatter_(1, action_idx.unsqueeze(1), 1.0)
        
        hidden_state, reward = network.dynamics(hidden_state, action_one_hot)
        
    assert not torch.isnan(hidden_state).any()
