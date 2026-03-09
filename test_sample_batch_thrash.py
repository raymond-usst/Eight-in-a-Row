import time
import pickle
import numpy as np
import os
import sys

from ai.replay_buffer import ReplayBuffer
from ai.replay_buffer import GameHistory

def test_thrash():
    os.makedirs("runs/replays/chunks", exist_ok=True)
    buffer = ReplayBuffer(max_size=50000, chunk_dir="runs/replays/chunks")
    buffer.chunk_size = 100
    buffer.max_cache_chunks = 10
    
    # create 3000 games to spam into the buffer, 30 chunks
    print("Generating games...")
    for j in range(3000):
        game = GameHistory()
        game.board_size = 30
        for i in range(10):
            game.actions.append(i % 10)
            game.rewards.append(0.0)
            game.policy_targets.append(np.zeros(21*21, dtype=np.float32))
            game.root_values.append(np.zeros(3, dtype=np.float32))
            game.threats.append(np.zeros(3, dtype=np.float32))
            game.player_ids.append((i % 3) + 1)
            game.centers.append((15, 15))
        game.winner = 1
        game.target_centers_precomputed = np.zeros((10, 2), dtype=np.int32)
        game.board_snapshots = {0: np.zeros((30, 30), dtype=np.int8)}
        buffer.save_game(game)
    
    print("Games saved. Starting sampling...")
    
    start = time.perf_counter()
    n_batches = 10
    for _ in range(n_batches):
        res = buffer.sample_batch(batch_size=128, num_unroll_steps=5, td_steps=5, discount=0.997, action_size=441, view_size=21)
    
    elapsed = time.perf_counter() - start
    print(f"Sampled {n_batches} batches in {elapsed:.3f}s, {elapsed/n_batches:.3f}s per batch ({elapsed*1000/(n_batches*128):.3f}ms per sample)")

if __name__ == "__main__":
    test_thrash()
