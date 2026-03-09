"""Standalone Arena Evaluation Script

This script pits two model checkpoints against each other in a series of games
to determine their relative strength (win rate and Elo difference).
"""

import os
import sys
import torch
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.self_play import play_game

def load_model(checkpoint_path, config, device):
    """Loads a MuZero model from a checkpoint."""
    model = MuZeroNetwork(config).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats (PBT agent vs standard network)
    if 'network' in checkpoint:
        state_dict = checkpoint['network']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Strip compiled prefix if necessary
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        cleaned_state_dict[k] = v
        
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model

def run_arena_match(model_a, model_b, config, num_games=20, temp=0.1, verbose=True):
    """
    Pits Model A against Model B for the specified number of games.
    Returns a dictionary of metrics including win rate and Elo difference.
    """
    if verbose:
        print(f"\nArena Match starts: Model A vs Model B ({num_games} games)")
        
    wins = {"A": 0, "B": 0, "Draw": 0}
    
    for i in range(num_games):
        play_a_as_p1 = (i % 2 == 0)
        
        if play_a_as_p1:
            networks = {1: model_a, 2: model_b, 3: model_a}
            a_pids = [1, 3]
            b_pids = [2]
        else:
            networks = {1: model_b, 2: model_a, 3: model_b}
            b_pids = [1, 3]
            a_pids = [2]

        if verbose:
            print(f"\nGame {i+1}/{num_games}: "
                  f"Model A is P{'+P'.join(map(str, a_pids))} | "
                  f"Model B is P{'+P'.join(map(str, b_pids))}")
              
        history = play_game(
            network=networks,
            config=config,
            temperature=temp,
            training_step=1000000, 
        )
        
        if history.winner is None:
            wins["Draw"] += 1
            if verbose: print(f"  Result: Draw (Length: {len(history)} moves)")
        elif history.winner in a_pids:
            wins["A"] += 1
            if verbose: print(f"  Result: Model A won! (Length: {len(history)} moves)")
        else:
            wins["B"] += 1
            if verbose: print(f"  Result: Model B won! (Length: {len(history)} moves)")
            
        if verbose:
            print(f"  Current Score: Model A: {wins['A']} | Model B: {wins['B']} | Draws: {wins['Draw']}")
        
    win_rate_b = (wins['B'] + 0.5 * wins['Draw']) / max(1, num_games)
    
    elo_diff = 0.0
    if 0.0 < win_rate_b < 1.0:
        elo_diff = -400 * np.log10(1.0 / win_rate_b - 1.0)
    elif win_rate_b == 1.0:
        elo_diff = 400.0  # Cap at +400
    else:
        elo_diff = -400.0 # Cap at -400

    if verbose:
        print("\n" + "="*40)
        print("Arena Evaluation Finished")
        print(f"Model A Wins: {wins['A']} | Model B Wins: {wins['B']} | Draws: {wins['Draw']}")
        print(f"Model B Win Rate: {win_rate_b * 100:.1f}%")
        print(f"Estimated Elo Difference (B vs A): {elo_diff:+.1f}")
        
    return {
        "wins_a": wins["A"],
        "wins_b": wins["B"],
        "draws": wins["Draw"],
        "win_rate_b": win_rate_b,
        "elo_diff": elo_diff
    }

def main():
    parser = argparse.ArgumentParser(description="AI Arena: Pit two models against each other.")
    parser.add_argument("--model_a", type=str, required=True, help="Path to checkpoint A (e.g., old model)")
    parser.add_argument("--model_b", type=str, required=True, help="Path to checkpoint B (e.g., new model)")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--board_size", type=int, default=15, help="Board size")
    parser.add_argument("--win_length", type=int, default=5, help="Win length constraint")
    parser.add_argument("--sims", type=int, default=50, help="MCTS Simulations per move")
    parser.add_argument("--temp", type=float, default=0.1, help="Sampling temperature")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Arena Evaluation on {device}")
    
    config = MuZeroConfig()
    config.n_layers = 12
    config.d_model = 256
    config.board_size = args.board_size
    config.win_length = args.win_length
    config.num_simulations_early = args.sims
    config.num_simulations_mid = args.sims
    config.num_simulations_late = args.sims
    
    model_a = load_model(args.model_a, config, device)
    model_b = load_model(args.model_b, config, device)
    
    run_arena_match(model_a, model_b, config, num_games=args.games, temp=args.temp, verbose=True)

if __name__ == '__main__':
    main()
