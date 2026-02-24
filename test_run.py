import torch
import numpy as np
from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.mcts import gumbel_muzero_search
import traceback

try:
    cfg = MuZeroConfig()
    cfg.board_size = 10
    cfg.local_view_size = 7
    cfg.policy_size = 49
    cfg.num_simulations = 4
    cfg.num_simulations_play = 4
    cfg.batch_size = 2

    net = MuZeroNetwork(cfg)
    net.eval()

    obs = np.random.randn(cfg.observation_channels, cfg.local_view_size, cfg.local_view_size).astype(np.float32)
    legal_mask = np.ones(cfg.policy_size, dtype=np.float32)

    action_probs, root_value, root = gumbel_muzero_search(
        net, obs, legal_mask, cfg,
        add_noise=False,
        session_context_vec=None,
        device=torch.device('cpu')
    )
    print("Action Probs Sum:", np.sum(action_probs))
    print("Root Value:", root_value)
    print("Visit Count:", root.visit_count if root else None)
    print("Success")
except Exception as e:
    traceback.print_exc()
