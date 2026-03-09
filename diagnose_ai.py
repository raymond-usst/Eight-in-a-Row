"""Diagnostic script for AI model quality.

Tests:
1. Does the model produce non-uniform policy distributions? (learned vs random)
2. Does the model correctly see the board state?
3. Does the model respond to obvious threats/opportunities?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.game_env import EightInARowEnv
from ai.mcts import gumbel_muzero_search, select_action

def load_model(path):
    config = MuZeroConfig()
    config.num_simulations_start = config.num_simulations_end = config.num_simulations_play
    config.validate()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MuZeroNetwork.from_config(config).to(device)
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
            network.load_state_dict(checkpoint['network_state_dict'], strict=False)
            step = checkpoint.get('training_step', '?')
            print(f"✓ Loaded checkpoint: {path}")
            print(f"  Training step: {step}")
            if 'curriculum_stage' in checkpoint:
                print(f"  Curriculum stage: {checkpoint['curriculum_stage']}")
        else:
            network.load_state_dict(checkpoint, strict=False)
            print(f"✓ Loaded weights: {path}")
    else:
        print(f"✗ File not found: {path}")
        return None, None, None
    
    network.eval()
    print(f"  Device: {device}, Parameters: {sum(p.numel() for p in network.parameters()):,}")
    return network, config, device


def test_empty_board(network, config, device):
    """Test 1: Does the model output a reasonable policy on an empty board?"""
    print("\n" + "="*60)
    print("TEST 1: Empty board policy")
    print("="*60)
    
    env = EightInARowEnv()
    env.reset()
    
    obs, center = env.get_observation(config.local_view_size)
    legal_mask = env.get_legal_actions_mask(center[0], center[1], config.local_view_size)
    
    # Raw network output (no MCTS)
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden, policy_logits, value = network.initial_inference(obs_tensor)
    
    logits = policy_logits.squeeze(0).cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(config.policy_size)
    
    print(f"  Policy entropy: {entropy:.2f} / {max_entropy:.2f} (ratio: {entropy/max_entropy:.2%})")
    print(f"  Value [me, next, prev]: {value[0].cpu().numpy()}")
    print(f"  Top 5 actions: {np.argsort(probs)[::-1][:5]} with probs: {np.sort(probs)[::-1][:5]}")
    
    if entropy / max_entropy > 0.95:
        print("  ⚠ Policy is nearly uniform — model may not have learned much")
    else:
        print("  ✓ Policy shows learned preferences")


def test_obvious_threat(network, config, device):
    """Test 2: Can the model find an obvious winning move (7 in a row)?"""
    print("\n" + "="*60)
    print("TEST 2: Obvious winning move (7 in a row, need to extend)")
    print("="*60)
    
    env = EightInARowEnv()
    env.reset()
    
    # Place 7 red pieces in a row at row 50
    for c in range(43, 50):
        env.board[50, c] = 1  # red
    # Add some opponent pieces nearby
    env.board[49, 45] = 2  # green
    env.board[51, 46] = 3  # blue
    
    # Reconstruct planes
    env._player_planes[0] = (env.board == 1).astype(np.float32)
    env._player_planes[1] = (env.board == 2).astype(np.float32)
    env._player_planes[2] = (env.board == 3).astype(np.float32)
    env._player_planes[3] = (env.board == 0).astype(np.float32)
    env._planes_dirty = True
    
    occupied = np.argwhere(env.board != 0)
    env._min_r = int(occupied[:, 0].min())
    env._max_r = int(occupied[:, 0].max())
    env._min_c = int(occupied[:, 1].min())
    env._max_c = int(occupied[:, 1].max())
    
    env.current_player = 0  # Red's turn
    
    obs, center = env.get_observation(config.local_view_size)
    legal_mask = env.get_legal_actions_mask(center[0], center[1], config.local_view_size)
    
    print(f"  Center: {center}")
    print(f"  Board state around row 50: {env.board[50, 40:55]}")
    
    # Check observation channels
    print(f"  Obs shape: {obs.shape}")
    print(f"  Local ch0 (me) sum: {obs[0].sum():.0f}")  # Should see 7 red pieces
    print(f"  Local ch1 (next) sum: {obs[1].sum():.0f}")  # green
    print(f"  Local ch2 (prev) sum: {obs[2].sum():.0f}")  # blue
    print(f"  Local ch3 (empty) sum: {obs[3].sum():.0f}")
    
    # Run MCTS with full simulations
    action_probs, root_value, root = gumbel_muzero_search(
        network, obs, legal_mask, config, add_noise=False,
        num_simulations_override=100
    )
    
    action = select_action(action_probs, temperature=0.1)
    board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)
    
    # The winning moves are (50, 42) or (50, 50) — extending either end
    winning_moves = [(50, 42), (50, 50)]
    
    print(f"\n  MCTS chose: ({board_r}, {board_c})")
    print(f"  Winning moves would be: {winning_moves}")
    print(f"  Root value: {root_value}")
    
    top_indices = np.argsort(action_probs)[::-1][:5]
    print(f"  Top 5 actions:")
    for idx in top_indices:
        r, c = env.action_to_board(int(idx), center[0], center[1], config.local_view_size)
        print(f"    ({r}, {c}) prob={action_probs[idx]:.3f}")
    
    if (board_r, board_c) in winning_moves:
        print("  ✓ Model found the winning move!")
    else:
        print("  ⚠ Model did NOT find the winning move")


def test_block_opponent(network, config, device):
    """Test 3: Can the model block an opponent's 7 in a row?"""
    print("\n" + "="*60)
    print("TEST 3: Block opponent's 7 in a row")
    print("="*60)
    
    env = EightInARowEnv()
    env.reset()
    
    # Green (player 2) has 7 in a row at row 50
    for c in range(43, 50):
        env.board[50, c] = 2  # green
    # Add some pieces
    env.board[48, 45] = 1  # red
    env.board[52, 46] = 3  # blue
    
    env._player_planes[0] = (env.board == 1).astype(np.float32)
    env._player_planes[1] = (env.board == 2).astype(np.float32)
    env._player_planes[2] = (env.board == 3).astype(np.float32)
    env._player_planes[3] = (env.board == 0).astype(np.float32)
    env._planes_dirty = True
    
    occupied = np.argwhere(env.board != 0)
    env._min_r = int(occupied[:, 0].min())
    env._max_r = int(occupied[:, 0].max())
    env._min_c = int(occupied[:, 1].min())
    env._max_c = int(occupied[:, 1].max())
    
    env.current_player = 0  # Red's turn — needs to block green
    
    obs, center = env.get_observation(config.local_view_size)
    legal_mask = env.get_legal_actions_mask(center[0], center[1], config.local_view_size)
    
    action_probs, root_value, root = gumbel_muzero_search(
        network, obs, legal_mask, config, add_noise=False,
        num_simulations_override=100
    )
    
    action = select_action(action_probs, temperature=0.1)
    board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)
    
    blocking_moves = [(50, 42), (50, 50)]
    
    print(f"  MCTS chose: ({board_r}, {board_c})")
    print(f"  Blocking moves would be: {blocking_moves}")
    print(f"  Root value: {root_value}")
    
    top_indices = np.argsort(action_probs)[::-1][:5]
    print(f"  Top 5 actions:")
    for idx in top_indices:
        r, c = env.action_to_board(int(idx), center[0], center[1], config.local_view_size)
        print(f"    ({r}, {c}) prob={action_probs[idx]:.3f}")
    
    if (board_r, board_c) in blocking_moves:
        print("  ✓ Model blocked the opponent!")
    else:
        print("  ⚠ Model did NOT block the opponent")


def test_checkpoint_info(path):
    """Show checkpoint metadata."""
    print("\n" + "="*60)
    print("CHECKPOINT INFO")
    print("="*60)
    
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        for key in sorted(checkpoint.keys()):
            if key == 'network_state_dict':
                print(f"  {key}: {len(checkpoint[key])} parameter tensors")
            elif key == 'optimizer_state_dict':
                print(f"  {key}: present")
            elif key == 'replay_buffer':
                print(f"  {key}: present")
            else:
                val = checkpoint[key]
                if isinstance(val, (int, float, str, bool, list, tuple)):
                    print(f"  {key}: {val}")
                elif isinstance(val, dict):
                    print(f"  {key}: dict with {len(val)} keys")
                else:
                    print(f"  {key}: {type(val).__name__}")
    else:
        print(f"  Raw state_dict with {len(checkpoint)} tensors")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_async/latest.pt'
    
    test_checkpoint_info(model_path)
    
    network, config, device = load_model(model_path)
    if network is None:
        sys.exit(1)
    
    test_empty_board(network, config, device)
    test_obvious_threat(network, config, device)
    test_block_opponent(network, config, device)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
