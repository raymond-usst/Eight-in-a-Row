"""Experience replay buffer for MuZero training.

Features:
- Memory-aware capacity management (configurable GB budget)
- Multi-factor quality scoring (outcome, length, recency, policy sharpness, model maturity)
- Tiered eviction (protects recent games, evicts lowest quality first)
- Prioritized experience replay (higher quality games sampled more frequently)
- System memory safety valve (emergency eviction when RAM critically low)
- Backward-compatible save/load with old format
"""

import numpy as np
import pickle
import os
import time
import threading
from typing import List, Tuple, Optional, Dict, Any

from ai.log_utils import get_logger
from ai.fast_board import replay_board_from_snapshot, build_local_obs

_log = get_logger(__name__)


class GameHistory:
    """Stores the trajectory of a single game."""

    SNAPSHOT_INTERVAL = 10  # Save board snapshot every N steps

    def __init__(self):
        self.actions: List[int] = []                     # local action indices
        self.rewards: List[float] = []                   # immediate rewards
        self.policy_targets: List[np.ndarray] = []       # MCTS visit distributions
        self.root_values: List[np.ndarray] = []          # MCTS root values (vector)
        self.threats: List[np.ndarray] = []              # Threat vectors [5, 6, 7+]
        self.player_ids: List[int] = []                  # who played each move
        self.centers: List[Tuple[int, int]] = []         # crop centers
        self.done: bool = False
        self.winner: Optional[int] = None
        # Ranking info (Phase 2): [(pid, placement_0indexed)] sorted 1st→3rd
        self.rankings: List[Tuple[int, int]] = []
        # Per-player placement reward: {pid: reward_float}
        self.placement_rewards: Dict[int, float] = {}
        # Session context (Phase 3-4): raw session-level data for context encoding
        self.session_scores: Optional[Dict[int, int]] = None  # {pid: cumulative_points} at game start
        self.session_game_idx: int = 0           # which game in the session (0-indexed)
        self.session_length: int = 1             # total games in session
        # Board snapshots for fast reconstruction in sample_batch (Opt 5)
        # {step_index: board_copy_int8_100x100}
        self.board_snapshots: Dict[int, np.ndarray] = {}
        # Precomputed focus data (computed once at save_game time)
        # target_centers_precomputed[i] = int index (row*100+col) of the actual move at step i
        self.target_centers_precomputed: Optional[List[int]] = None
        
        # Phase 3: Curriculum Learning Support
        self.board_size: int = 100
        self.win_length: int = 8
        # Final board state (for visualization); set at end of self-play
        self.final_board: Optional[np.ndarray] = None

    def store(self, action: int, reward: float,
              policy: np.ndarray, root_value: np.ndarray, threats: np.ndarray,
              player_id: int, center: Tuple[int, int]):
        """Append one step: action, reward, policy target, root value, threats, player_id, center."""
        self.actions.append(action)
        self.rewards.append(reward)
        self.policy_targets.append(policy)
        self.root_values.append(root_value)
        self.threats.append(threats)
        self.player_ids.append(player_id)
        self.centers.append(center)

    def __len__(self):
        return len(self.actions)


class ReplayBuffer:
    """Memory-aware replay buffer with quality-based retention and prioritized sampling.

    Instead of a simple FIFO circular buffer, this implementation:
    1. Tracks estimated memory per game and enforces a configurable GB budget
    2. Scores each game's quality based on multiple factors
    3. When over budget, batch-evicts lowest-quality older games (protecting recent ones)
    4. Samples games proportional to quality^alpha (prioritized experience replay)
    5. Monitors system RAM as emergency safety valve

    Elasticity: Growth is bounded by max_size and max_memory_gb. Eviction runs when
    total_memory or len(games) exceeds budget. sample_batch() requires at least one
    game (raises ValueError with num_games=0 otherwise). Producers (e.g. learner
    receiving games) may block on full queue until consumer drains.
    """

    # ── Quality scoring weight factors ──
    W_OUTCOME   = 0.25   # Decisive win vs draw
    W_LENGTH    = 0.20   # Game length in optimal range
    W_RECENCY   = 0.25   # How recently the game was generated
    W_SHARPNESS = 0.15   # MCTS policy confidence (low entropy)
    W_MATURITY  = 0.15   # Training step of the model that generated the game

    # ── Length scoring parameters ──
    OPTIMAL_LEN_MIN = 30
    OPTIMAL_LEN_MAX = 120
    MIN_USEFUL_LEN  = 10
    MAX_USEFUL_LEN  = 250

    # ── Eviction parameters ──
    EVICTION_TARGET_RATIO  = 0.85  # Evict down to 85% of budget (hysteresis)
    RECENT_PROTECT_RATIO   = 0.20  # Protect newest 20% from eviction
    SYSTEM_RAM_EMERGENCY_GB = 3.0  # Emergency eviction if available RAM < this
    # Scalability: skip O(n) num_positions() in memory_report when game count exceeds this
    _REPORT_NUM_POSITIONS_MAX_GAMES = 5000

    def __init__(self, max_size: int = 50000, max_memory_gb: float = 35.0,
                 priority_alpha: float = 0.6, min_games: int = 100, chunk_dir: str = "runs/replays/chunks"):
        """
        Args:
            max_size: Maximum number of games (hard cap regardless of memory)
            max_memory_gb: Memory budget in GB for the buffer contents
            priority_alpha: Sampling priority sharpness (0=uniform, 1=fully proportional)
            min_games: Minimum games to keep (won't evict below this)
            chunk_dir: Directory to save chunked game files.
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_gb * (1024 ** 3))
        self.priority_alpha = priority_alpha
        self.min_games = min_games
        self.chunk_dir = chunk_dir
        os.makedirs(self.chunk_dir, exist_ok=True)

        # Storage (Chunked)
        self.active_chunk: List[GameHistory] = []
        self.chunk_cache: Dict[int, List[GameHistory]] = {}
        self.max_cache_chunks = max(10, (self.max_size // 1000) * 2)  # Enough to cache the entire buffer
        self.chunk_size = 1000
        self.chunk_id_counter = 0

        self.meta: List[Dict[str, Any]] = []

        # Tracking
        self.total_memory = 0          # Current estimated memory usage (bytes)
        self.total_games = 0           # Monotonic counter of all games ever added
        self._eviction_count = 0       # Total games evicted across lifetime

        # ── Vectorized sampling arrays (pre-allocated with capacity doubling) ──
        # Avoids O(n) np.append copies. Uses _arr_len to track logical size.
        self._arr_capacity = 1024
        self._arr_len = 0
        self._quality_arr = np.empty(self._arr_capacity, dtype=np.float64)
        self._gamelen_arr = np.empty(self._arr_capacity, dtype=np.float64)
        # Cached sampling weights (invalidated on any mutation)
        self._sampling_weights: Optional[np.ndarray] = None
        self._weights_dirty = True

        # Active chunk version counter for lock-free snapshot
        self._active_chunk_version = 0
        self._active_chunk_snapshot = []
        self._active_chunk_snapshot_version = -1

        # Timing instrumentation for sample_batch
        self._batch_call_count = 0
        self._batch_timing = {'lock': 0.0, 'sample': 0.0, 'total': 0.0}

        # Thread safety: protects self.meta and active_chunk against concurrent accesses
        self._data_lock = threading.Lock()

    def clear(self):
        """Reset the buffer to empty (for curriculum transitions)."""
        with self._data_lock:
            self.active_chunk = []
            self.chunk_cache.clear()
            self.meta = []
            self.total_memory = 0
            self._quality_arr = np.empty(0, dtype=np.float64)
            self._gamelen_arr = np.empty(0, dtype=np.float64)
            self._sampling_weights = None
            self._weights_dirty = True
            
            # Clean disk
            for f in os.listdir(self.chunk_dir):
                if f.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(self.chunk_dir, f))
                    except OSError:
                        pass
            
            print("[ReplayBuffer] Buffer cleared.")

    # ================================================================
    #  Memory Estimation
    # ================================================================

    @staticmethod
    def _estimate_game_memory(game: GameHistory) -> int:
        """Estimate memory footprint of a GameHistory in bytes."""
        n = len(game)
        if n == 0:
            return 256  # Minimal empty-object overhead

        # Numpy array sizes — use .nbytes directly on arrays (O(1) instead of summing elements)
        pt = game.policy_targets
        if isinstance(pt, np.ndarray):
            policy_bytes = pt.nbytes
        elif len(pt) > 0:
            policy_bytes = sum(p.nbytes for p in pt)
        else:
            policy_bytes = 0

        # Board snapshots (100x100 int8 = 10KB each)
        snapshots = getattr(game, 'board_snapshots', {})
        snapshot_bytes = sum(s.nbytes for s in snapshots.values()) if snapshots else 0

        target_centers = getattr(game, 'target_centers_precomputed', None)
        if target_centers is not None:
            if isinstance(target_centers, np.ndarray):
                precomputed_bytes = target_centers.nbytes
            elif len(target_centers) > 0:
                precomputed_bytes = len(target_centers) * 8
            else:
                precomputed_bytes = 0
        else:
            precomputed_bytes = 0

        # Per-step overhead for remaining arrays (actions, rewards, root_values, etc.)
        # After _save_game_locked converts to numpy, these are compact.
        per_step = 40  # conservative estimate for numpy-backed storage
        scalar_bytes = n * per_step

        # Base object overhead + metadata fields
        base_overhead = 1280

        return policy_bytes + snapshot_bytes + precomputed_bytes + scalar_bytes + base_overhead

    # ================================================================
    #  Quality Scoring
    # ================================================================

    def _compute_quality(self, game: GameHistory, meta: Dict) -> float:
        """Compute multi-factor quality score in [0, 1]."""
        score = 0.0
        game_len = len(game)

        # ── 1. Outcome: decisive wins are more informative ──
        outcome_s = 1.0 if game.winner is not None else 0.3
        score += self.W_OUTCOME * outcome_s

        # ── 2. Length: bell-curve around optimal range ──
        if game_len < self.MIN_USEFUL_LEN:
            length_s = 0.1
        elif game_len < self.OPTIMAL_LEN_MIN:
            t = (game_len - self.MIN_USEFUL_LEN) / max(1, self.OPTIMAL_LEN_MIN - self.MIN_USEFUL_LEN)
            length_s = 0.3 + 0.7 * t
        elif game_len <= self.OPTIMAL_LEN_MAX:
            length_s = 1.0
        elif game_len <= self.MAX_USEFUL_LEN:
            t = (game_len - self.OPTIMAL_LEN_MAX) / max(1, self.MAX_USEFUL_LEN - self.OPTIMAL_LEN_MAX)
            length_s = 1.0 - 0.6 * t
        else:
            length_s = 0.2
        score += self.W_LENGTH * length_s

        # ── 3. Recency: exponential decay (half-life ≈ 3500 games) ──
        age = max(0, self.total_games - meta.get('insert_idx', 0))
        recency_s = np.exp(-age / 5000.0)
        score += self.W_RECENCY * recency_s

        # ── 4. Policy sharpness: low entropy = decisive MCTS (vectorized) ──
        sharpness_s = 0.5  # Default if can't compute
        if len(game.policy_targets) > 0 and game_len > 0:
            try:
                sample_idx = np.linspace(0, game_len - 1, min(5, game_len), dtype=int)
                # Batch: stack sampled policies into (K, A) matrix
                if isinstance(game.policy_targets, np.ndarray):
                    sampled = game.policy_targets[sample_idx].astype(np.float64)
                else:
                    sampled = np.array([game.policy_targets[i] for i in sample_idx], dtype=np.float64)
                sampled = np.nan_to_num(sampled, nan=1e-12, posinf=1.0, neginf=1e-12)
                sampled = np.clip(sampled, 1e-12, 1.0)
                # Vectorized entropy: -sum(p * log(p)) per row
                entropies = -np.sum(np.where(sampled > 0, sampled * np.log(sampled), 0.0), axis=1)
                avg_entropy = entropies.mean()
                max_entropy = np.log(sampled.shape[1])  # log(441) ≈ 6.09
                sharpness_s = max(0.0, 1.0 - avg_entropy / max_entropy) if max_entropy > 0 else 0.5
            except Exception:
                pass
        score += self.W_SHARPNESS * sharpness_s

        # ── 5. Model maturity: games from later training steps are better ──
        step = meta.get('training_step', 0)
        maturity_s = min(1.0, step / 50000.0)
        score += self.W_MATURITY * maturity_s

        return score

    # ================================================================
    #  Game Insertion & Eviction
    # ================================================================

    @staticmethod
    def _precompute_focus_data(game: GameHistory):
        """Precompute target_centers for every position (vectorized).
        Called once at save_game() time.
        """
        BOARD_SIZE = getattr(game, 'board_size', 100)
        VIEW_SIZE = 21
        HALF = VIEW_SIZE // 2
        n = len(game)
        if n == 0:
            game.target_centers_precomputed = np.array([], dtype=np.int32)
            return

        # Vectorized computation — no Python loop
        actions = np.asarray(game.actions, dtype=np.int32)
        centers = np.asarray(game.centers, dtype=np.int32)  # (n, 2)
        lr = actions // VIEW_SIZE
        lc = actions % VIEW_SIZE
        tr = np.clip(centers[:, 0] - HALF + lr, 0, BOARD_SIZE - 1)
        tc = np.clip(centers[:, 1] - HALF + lc, 0, BOARD_SIZE - 1)
        game.target_centers_precomputed = (tr * BOARD_SIZE + tc).astype(np.int32)

    def save_game(self, game: GameHistory, training_step: int = 0):
        """Add a completed game with quality scoring and memory-aware eviction.
        Thread-safe: acquires _data_lock to protect games/meta lists."""
        # Precompute focus target centers BEFORE locking
        if game.target_centers_precomputed is None:
            self._precompute_focus_data(game)
        with self._data_lock:
            self._save_game_locked(game, training_step)

    def _save_game_locked(self, game: GameHistory, training_step: int = 0):
        # 1. Compress game lists into minimal numpy arrays identically to save space
        if len(game.actions) > 0 and isinstance(game.actions, list):
            game.actions = np.array(game.actions, dtype=np.int32)
            game.rewards = np.array(game.rewards, dtype=np.float16)
            game.policy_targets = np.array(game.policy_targets, dtype=np.float16)
            game.root_values = np.array(game.root_values, dtype=np.float16)
            game.threats = np.array(game.threats, dtype=np.float16)
            game.player_ids = np.array(game.player_ids, dtype=np.int8)
            if not isinstance(game.centers, np.ndarray):
                game.centers = np.array(game.centers, dtype=np.int32)
            if not isinstance(game.target_centers_precomputed, np.ndarray):
                game.target_centers_precomputed = np.array(game.target_centers_precomputed, dtype=np.int32)

        # Ensure step-0 board snapshot exists (avoids fallback to replay_start=0 in sample_batch)
        if 0 not in game.board_snapshots:
            bs = getattr(game, 'board_size', 100)
            game.board_snapshots[0] = np.zeros((bs, bs), dtype=np.int8)

        mem = self._estimate_game_memory(game)
        
        # Meta tracks where this game lives
        chunk_idx = len(self.active_chunk)
        chunk_id = self.chunk_id_counter
        game_len = len(game)
        
        meta = {
            'chunk_id': chunk_id,
            'chunk_idx': chunk_idx,
            'insert_idx': self.total_games,
            'training_step': training_step,
            'memory_bytes': mem,
            'quality': 0.0,
            'timestamp': time.time(),
            'game_len': game_len,
        }
        meta['quality'] = self._compute_quality(game, meta)

        self.total_games += 1
        self.active_chunk.append(game)
        self.meta.append(meta)
        self.total_memory += mem
        self._weights_dirty = True
        self._active_chunk_version += 1
        
        # O(1) amortized insertion into pre-allocated arrays (no np.append copy!)
        idx = self._arr_len
        if idx >= self._arr_capacity:
            new_cap = self._arr_capacity * 2
            new_q = np.empty(new_cap, dtype=np.float64)
            new_g = np.empty(new_cap, dtype=np.float64)
            new_q[:idx] = self._quality_arr[:idx]
            new_g[:idx] = self._gamelen_arr[:idx]
            self._quality_arr = new_q
            self._gamelen_arr = new_g
            self._arr_capacity = new_cap
        self._quality_arr[idx] = meta['quality']
        self._gamelen_arr[idx] = float(game_len)
        self._arr_len = idx + 1
        
        # Flush chunk to disk if full
        if len(self.active_chunk) >= self.chunk_size:
            chunk_path = os.path.join(self.chunk_dir, f'chunk_{chunk_id}.pkl')
            try:
                with open(chunk_path, 'wb') as f:
                    pickle.dump(self.active_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Cache it so immediate sampling is fast
                self._add_to_cache(chunk_id, self.active_chunk)
                self.active_chunk = []
                self.chunk_id_counter += 1
            except Exception as e:
                _log.error("Failed to write chunk %d: %s", chunk_id, e)

        # Evict if over budget
        self._maybe_evict()
        
    def _add_to_cache(self, chunk_id: int, games: List[GameHistory]):
        """LRU cache insertion for a chunk."""
        self.chunk_cache[chunk_id] = games
        if len(self.chunk_cache) > self.max_cache_chunks:
            # Evict oldest (simplest LRU since python 3.7 dicts preserve insertion order)
            oldest = next(iter(self.chunk_cache))
            del self.chunk_cache[oldest]

    def _maybe_evict(self):
        """Check if eviction is needed and run it."""
        needs_eviction = (
            self.total_memory > self.max_memory_bytes or
            len(self.meta) > self.max_size
        )
        if needs_eviction:
            target_mem = int(self.max_memory_bytes * self.EVICTION_TARGET_RATIO)
            target_cnt = int(self.max_size * 0.90)
            reason = "memory" if self.total_memory > self.max_memory_bytes else "count"
            self._run_eviction(target_mem, target_cnt, reason=reason)
        else:
            # Safety valve: check system memory
            self._check_system_memory()

    def _run_eviction(self, target_memory: int, target_count: int, reason: str = "memory"):
        """Batch-evict lowest-quality games to reach target levels.
        Identifies unreferenced chunks and deletes them from disk.
        """
        n = len(self.meta)
        if n <= self.min_games:
            return

        # Number of protected recent games
        n_protected = max(self.min_games, int(n * self.RECENT_PROTECT_RATIO))
        n_candidates = n - n_protected
        if n_candidates <= 0:
            return

        # Re-score candidates — vectorized recency update
        insert_idxs = np.array([self.meta[i].get('insert_idx', 0) for i in range(n_candidates)], dtype=np.float64)
        ages = np.maximum(0, self.total_games - insert_idxs)
        recency_new = np.exp(-ages / 5000.0)
        old_ages = np.maximum(0, self.total_games - 1 - insert_idxs)
        recency_old = np.exp(-old_ages / 5000.0)
        
        scored = []
        for i in range(n_candidates):
            base_quality = self.meta[i]['quality'] - self.W_RECENCY * recency_old[i]
            new_quality = base_quality + self.W_RECENCY * recency_new[i]
            self.meta[i]['quality'] = new_quality
            self._quality_arr[i] = new_quality
            scored.append((i, new_quality, self.meta[i]['memory_bytes']))

        # Sort by quality ascending (worst first)
        scored.sort(key=lambda x: x[1])

        # Collect indices to evict
        evict_set = set()
        freed = 0
        for idx, quality, mem in scored:
            remaining_n = n - len(evict_set)
            remaining_mem = self.total_memory - freed

            if remaining_mem <= target_memory and remaining_n <= target_count:
                break
            if remaining_n <= self.min_games:
                break

            evict_set.add(idx)
            freed += mem

        if not evict_set:
            return

        # Rebuild meta lists
        new_meta = []
        active_chunk_ids = set()
        for i in range(n):
            if i not in evict_set:
                new_meta.append(self.meta[i])
                active_chunk_ids.add(self.meta[i]['chunk_id'])

        # Identify deleted chunks
        all_chunk_ids = {m['chunk_id'] for m in self.meta}
        deleted_chunks = all_chunk_ids - active_chunk_ids

        # Delete unused chunk files from disk
        for cid in deleted_chunks:
            # If it's the active chunk (not yet saved), don't delete file, just remove from list if 0
            if cid == self.chunk_id_counter:
                # Active chunk games were evicted! Rebuild active_chunk
                valid_active = []
                for g_meta in new_meta:
                    if g_meta['chunk_id'] == cid:
                        valid_active.append(self.active_chunk[g_meta['chunk_idx']])
                self.active_chunk = valid_active
                # Reindex
                for g_meta in new_meta:
                    if g_meta['chunk_id'] == cid:
                        # Find its new index
                        pass # Actually this requires O(n) scan. Let's just avoid evicting active chunk if possible.
            else:
                chunk_path = os.path.join(self.chunk_dir, f'chunk_{cid}.pkl')
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except OSError:
                    pass
                # Remove from cache if present
                self.chunk_cache.pop(cid, None)

        evicted = len(evict_set)
        self._eviction_count += evicted
        self.meta = new_meta
        self.total_memory -= freed
        self._weights_dirty = True
        
        # Rebuild vectorized arrays from surviving meta
        new_n = len(new_meta)
        self._quality_arr = np.empty(max(new_n * 2, 1024), dtype=np.float64)
        self._gamelen_arr = np.empty(max(new_n * 2, 1024), dtype=np.float64)
        self._arr_capacity = max(new_n * 2, 1024)
        self._arr_len = new_n
        for i, m in enumerate(new_meta):
            self._quality_arr[i] = m['quality']
            self._gamelen_arr[i] = m.get('game_len', 1)

        avg_quality_evicted = np.mean([scored[j][1] for j in range(evicted)]) if evicted > 0 else 0
        avg_quality_kept = np.mean([m['quality'] for m in self.meta]) if self.meta else 0
        print(f"[ReplayBuffer] Evicted {evicted} games (reason: {reason}, "
              f"avg_q={avg_quality_evicted:.3f}, freed {freed / (1024**2):.0f} MB). "
              f"Kept {len(self.meta)} games "
              f"(avg_q={avg_quality_kept:.3f}, {self.total_memory / (1024**3):.2f} GB)",
              flush=True)

    def _check_system_memory(self):
        """Emergency eviction if system RAM is critically low."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < self.SYSTEM_RAM_EMERGENCY_GB and len(self.games) > self.min_games:
                _log.warning("EMERGENCY: System RAM low (%.1f GB). Emergency eviction...", available_gb)
                target_mem = int(self.total_memory * 0.70)
                target_cnt = int(len(self.games) * 0.70)
                self._run_eviction(target_mem, target_cnt, reason="emergency")
        except ImportError:
            pass  # psutil not available, rely on estimation-based limits

    # ================================================================
    #  Sampling
    # ================================================================

    def _get_sampling_weights(self) -> np.ndarray:
        """Compute normalized sampling weights for prioritized replay (vectorized, O(1)-like)."""
        n = len(self.meta)
        if n == 0:
            return np.array([], dtype=np.float64)

        if not self._weights_dirty and self._sampling_weights is not None:
            if len(self._sampling_weights) == n:
                return self._sampling_weights

        # Vectorized: use parallel numpy arrays instead of Python loop
        q = np.maximum(self._quality_arr[:n], 0.01)
        gl = np.maximum(self._gamelen_arr[:n], 1.0)
        weights = np.power(q, self.priority_alpha) * np.sqrt(gl)

        # Stability: avoid division by zero
        total = weights.sum()
        if total > 0:
            weights *= (1.0 / max(total, 1e-9))
        else:
            weights[:] = 1.0 / n

        self._sampling_weights = weights
        self._weights_dirty = False
        return weights

    def sample_batch(self, batch_size: int, num_unroll_steps: int,
                     td_steps: int, discount: float,
                     action_size: int, view_size: int = 21) -> Dict[str, np.ndarray]:
        """
        Sample a batch of positions with prioritized experience replay.
        """
        t_total_start = time.perf_counter()
        # Snapshot metadata under lock — avoid O(n) list copy
        t_lock_start = time.perf_counter()
        with self._data_lock:
            n = len(self.meta)
            if n == 0:
                raise ValueError(
                "Cannot sample from empty replay buffer (num_games=0). "
                "Ensure buffer has games before calling sample_batch."
            )
            weights = self._get_sampling_weights()
            # Lightweight snapshot: reference the list (meta dicts are append-only;
            # only 'quality' mutates but that's safe for sampling purposes)
            meta_ref = self.meta
            meta_len = n
            # Fast active chunk snapshot with version check (avoids list() copy when unchanged)
            if self._active_chunk_snapshot_version != self._active_chunk_version:
                self._active_chunk_snapshot = list(self.active_chunk)
                self._active_chunk_snapshot_version = self._active_chunk_version
            active_chunk_snapshot = self._active_chunk_snapshot
            current_chunk_id = self.chunk_id_counter
        t_lock_end = time.perf_counter()

        batch_obs, batch_next_obs, batch_actions, batch_target_values = [], [], [], []
        batch_target_rewards, batch_target_policies, batch_global_states = [], [], []
        batch_target_centers, batch_session_contexts, batch_threats = [], [], []
        batch_heatmaps, batch_opponent_actions, batch_player_ids, batch_insert_idxs = [], [], [], []

        # Prioritized game selection
        game_indices = np.random.choice(n, size=batch_size, p=weights, replace=True)

        for gi in game_indices:
            if gi >= meta_len:
                continue
            game_meta = meta_ref[gi]
            cid = game_meta['chunk_id']
            cidx = game_meta['chunk_idx']
            
            # Fetch game object
            if cid == current_chunk_id:
                if cidx < len(active_chunk_snapshot):
                    game = active_chunk_snapshot[cidx]
                else:
                    continue
            else:
                if cid not in self.chunk_cache:
                    chunk_path = os.path.join(self.chunk_dir, f'chunk_{cid}.pkl')
                    try:
                        with open(chunk_path, 'rb') as f:
                            chunk_games = pickle.load(f)
                        with self._data_lock:
                            self._add_to_cache(cid, chunk_games)
                    except Exception as e:
                        _log.error("Failed to load chunk %d during sampling: %s", cid, e)
                        continue
                if cid in self.chunk_cache and cidx < len(self.chunk_cache[cid]):
                    game = self.chunk_cache[cid][cidx]
                else:
                    continue

            if len(game) == 0:
                continue
            pos = np.random.randint(len(game))

            # Provide a dummy observation here, we will replace this block using reconstruction logic
            # below after the board is reconstructed.
            batch_obs.append(None)
            batch_next_obs.append(None)

            # Unroll targets
            actions = []
            target_values = []
            target_rewards = []
            target_policies = []

            # Value target for initial position
            target_values.append(
                self._compute_value_target(game, pos, td_steps, discount)
            )
            target_policies.append(game.policy_targets[pos])

            for step in range(num_unroll_steps):
                idx = pos + step
                if idx < len(game) - 1:
                    actions.append(game.actions[idx])
                    target_rewards.append(game.rewards[idx])
                    target_values.append(
                        self._compute_value_target(game, idx + 1, td_steps, discount)
                    )
                    target_policies.append(
                        game.policy_targets[idx + 1] if idx + 1 < len(game)
                        else np.zeros(action_size, dtype=np.float32)
                    )
                else:
                    actions.append(0)
                    target_rewards.append(0.0)
                    target_values.append(np.zeros(3, dtype=np.float32))
                    target_policies.append(np.zeros(action_size, dtype=np.float32))

            batch_actions.append(actions)
            batch_target_values.append(target_values)
            batch_target_rewards.append(target_rewards)
            batch_target_policies.append(target_policies)

            # --- Focus Network Data (sparse snapshot + short replay) ---
            BOARD_SIZE = getattr(game, 'board_size', 100)
            VIEW_SIZE = view_size
            HALF = VIEW_SIZE // 2
            
            # Find nearest precomputed snapshot and replay at most ~SNAP_INTERVAL steps
            snap_interval = GameHistory.SNAPSHOT_INTERVAL
            snap_step = (pos // snap_interval) * snap_interval

            if snap_step in game.board_snapshots:
                current_board = game.board_snapshots[snap_step].copy()
                replay_start = snap_step
            elif 0 in game.board_snapshots:
                current_board = game.board_snapshots[0].copy()
                replay_start = 0
            else:
                current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
                replay_start = 0

            # C-accelerated board replay from snapshot to pos
            if replay_start < pos:
                _actions = game.actions if isinstance(game.actions, np.ndarray) else np.asarray(game.actions)
                _pids = game.player_ids if isinstance(game.player_ids, np.ndarray) else np.asarray(game.player_ids)
                if hasattr(game, '_centers_r'):
                    _centers_r = game._centers_r
                    _centers_c = game._centers_c
                else:
                    _centers_arr = np.asarray(game.centers)
                    _centers_r = _centers_arr[:, 0] if _centers_arr.ndim == 2 else np.array([c[0] for c in game.centers])
                    _centers_c = _centers_arr[:, 1] if _centers_arr.ndim == 2 else np.array([c[1] for c in game.centers])
                replay_board_from_snapshot(
                    current_board, _actions, _centers_r, _centers_c, _pids,
                    replay_start, pos, VIEW_SIZE, BOARD_SIZE
                )

            # --- Dynamic Sub-Observation Reconstruction (C-accelerated) ---
            current_pid = game.player_ids[pos]
            next_pid = (current_pid % 3) + 1
            prev_pid = ((current_pid + 1) % 3) + 1
            
            cpos = game.centers[pos]
            obs_local = build_local_obs(
                current_board, cpos[0], cpos[1],
                current_pid, next_pid, prev_pid, VIEW_SIZE, BOARD_SIZE
            )
            
            # Global thumbnail
            rotated_planes = {
                1: (current_board == current_pid),
                2: (current_board == next_pid),
                3: (current_board == prev_pid),
                0: (current_board == 0),
            }
            from ai.game_env import EightInARowEnv
            if not hasattr(self, '_obs_env') or self._obs_env.BOARD_SIZE != BOARD_SIZE:
                self._obs_env = EightInARowEnv(board_size=BOARD_SIZE)
            rot_stack = np.stack([rotated_planes[1], rotated_planes[2], rotated_planes[3], rotated_planes[0]], axis=0).astype(np.float32)
            obs_global = self._obs_env._numpy_area_pool(rot_stack, VIEW_SIZE)
            obs_array = np.concatenate([obs_local, obs_global], axis=0)
            
            # Replace the dummy None we pushed earlier
            batch_obs[-1] = obs_array
            
            # Repeat reconstruction for `pos + 1` (next_obs) if not terminal
            if pos + 1 < len(game):
                # Apply move `pos` to global board
                act = game.actions[pos]
                ctr = game.centers[pos]
                lr = act // VIEW_SIZE
                lc = act % VIEW_SIZE
                br = ctr[0] - HALF + lr
                bc = ctr[1] - HALF + lc
                if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                    current_board[br, bc] = current_pid
                
                # Next player context
                next_active_pid = game.player_ids[pos + 1]
                n_next_pid = (next_active_pid % 3) + 1
                n_prev_pid = ((next_active_pid + 1) % 3) + 1
                
                # C-accelerated local obs construction
                cpos_next = game.centers[pos + 1]
                obs_local_next = build_local_obs(
                    current_board, cpos_next[0], cpos_next[1],
                    next_active_pid, n_next_pid, n_prev_pid, VIEW_SIZE, BOARD_SIZE
                )
                
                rotated_planes_next = {
                    1: (current_board == next_active_pid),
                    2: (current_board == n_next_pid),
                    3: (current_board == n_prev_pid),
                    0: (current_board == 0),
                }
                rot_stack_next = np.stack([rotated_planes_next[1], rotated_planes_next[2], rotated_planes_next[3], rotated_planes_next[0]], axis=0).astype(np.float32)
                obs_global_next = self._obs_env._numpy_area_pool(rot_stack_next, VIEW_SIZE)
                obs_next_array = np.concatenate([obs_local_next, obs_global_next], axis=0)
                
                batch_next_obs[-1] = obs_next_array
            else:
                batch_next_obs[-1] = obs_array

            # Efficiency: vectorized global state (no per-cell Python loop).

            glob_state = np.stack([
                (current_board == current_pid),
                (current_board == next_pid),
                (current_board == prev_pid),
                (current_board == 0),
            ]).astype(np.float32)
            batch_global_states.append(glob_state)

            # Target center (precomputed for every position)
            batch_target_centers.append(game.target_centers_precomputed[pos])

            # --- Session Context (Phase 4) ---
            # Compute per-position 4-dim context vector from game-level session data
            if game.session_scores is not None:
                pid = game.player_ids[pos]
                max_possible = max(1, game.session_length * 5)
                pids_sorted = sorted(game.session_scores.keys())
                others = [p for p in pids_sorted if p != pid]
                my_score = game.session_scores.get(pid, 0) / max_possible
                opp1_score = game.session_scores.get(others[0], 0) / max_possible if len(others) > 0 else 0.0
                opp2_score = game.session_scores.get(others[1], 0) / max_possible if len(others) > 1 else 0.0
                games_remaining = (game.session_length - game.session_game_idx - 1) / max(1, game.session_length)
                ctx = np.array([my_score, opp1_score, opp2_score, games_remaining], dtype=np.float32)
            else:
                ctx = np.zeros(4, dtype=np.float32)
            batch_session_contexts.append(ctx)
            
            # --- Auxiliary Targets (Phase 2) ---
            # 1. Threats
            # Pull threat vector from current position. Use zeros if missing (legacy)
            if hasattr(game, 'threats') and pos < len(game.threats):
                batch_threats.append(game.threats[pos])
            else:
                batch_threats.append(np.zeros(3, dtype=np.float32))

            # 2. Opponent Action (Future Action)
            # Predict action at pos+1. If terminal, use -1 (masked later) or 0? 
            # We'll valid mask it generally, but here let's store index.
            if pos + 1 < len(game):
                batch_opponent_actions.append(game.actions[pos+1])
            else:
                batch_opponent_actions.append(-1) # Terminal

            # 3. Board Heatmap (Next N=20 steps) — vectorized
            heatmap = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.float32)
            lookahead = 20
            start_idx = pos + 1
            end_idx = min(len(game), start_idx + lookahead)

            if end_idx > start_idx:
                ctr = game.centers[pos]
                h_half = VIEW_SIZE // 2

                # Extract future actions and centers as arrays
                future_actions = np.array(game.actions[start_idx:end_idx], dtype=np.int32)
                future_centers = np.array(game.centers[start_idx:end_idx], dtype=np.int32)  # (N, 2)

                # Compute absolute board coordinates from local action + center
                lr = future_actions // VIEW_SIZE
                lc = future_actions % VIEW_SIZE
                abs_r = future_centers[:, 0] - h_half + lr
                abs_c = future_centers[:, 1] - h_half + lc

                # Convert to relative coordinates w.r.t. current view center
                rel_r = abs_r - (ctr[0] - h_half)
                rel_c = abs_c - (ctr[1] - h_half)

                # Mask: keep only coordinates within [0, VIEW_SIZE); clip to avoid negative indices (numpy forbids them in fancy indexing)
                valid = (rel_r >= 0) & (rel_r < VIEW_SIZE) & (rel_c >= 0) & (rel_c < VIEW_SIZE)
                rel_r_safe = np.clip(rel_r, 0, VIEW_SIZE - 1).astype(np.intp)
                rel_c_safe = np.clip(rel_c, 0, VIEW_SIZE - 1).astype(np.intp)
                heatmap[rel_r_safe[valid], rel_c_safe[valid]] = 1.0
            
            batch_heatmaps.append(heatmap)
            
            # KOTH Support: Add player_id of current move
            batch_player_ids.append(game.player_ids[pos])
            batch_insert_idxs.append(game_meta.get('insert_idx', -1))

        result = {
            'observations': np.array(batch_obs, dtype=np.float32),
            'next_observations': np.array(batch_next_obs, dtype=np.float32),
            'actions': np.array(batch_actions, dtype=np.int64),
            'target_values': np.array(batch_target_values, dtype=np.float32),
            'target_rewards': np.array(batch_target_rewards, dtype=np.float32),
            'target_policies': np.array(batch_target_policies, dtype=np.float32),
            'global_states': np.array(batch_global_states, dtype=np.float32),
            'target_centers': np.array(batch_target_centers, dtype=np.int64),
            'session_contexts': np.array(batch_session_contexts, dtype=np.float32),
            'target_threats': np.array(batch_threats, dtype=np.float32),
            'target_opponent_actions': np.array(batch_opponent_actions, dtype=np.int64),
            'target_heatmaps': np.array(batch_heatmaps, dtype=np.float32),
            'player_ids': np.array(batch_player_ids, dtype=np.int64),
            'insert_idxs': np.array(batch_insert_idxs, dtype=np.int64),
        }

        # Timing instrumentation
        t_total_end = time.perf_counter()
        self._batch_call_count += 1
        self._batch_timing['lock'] += t_lock_end - t_lock_start
        self._batch_timing['total'] += t_total_end - t_total_start
        if self._batch_call_count % 50 == 0:
            avg_total = self._batch_timing['total'] / self._batch_call_count
            avg_lock = self._batch_timing['lock'] / self._batch_call_count
            print(f"[ReplayBuffer] Batch timing (avg over {self._batch_call_count} calls): "
                  f"total={avg_total:.3f}s, lock={avg_lock:.4f}s, "
                  f"per_sample={avg_total/batch_size*1000:.1f}ms, "
                  f"games={len(self.meta)}",
                  flush=True)

        return result

    def update_priorities(self, insert_idxs: np.ndarray, errors: np.ndarray):
        """Update quality score of games based on TD errors from training."""
        with self._data_lock:
            max_errs = {}
            for idx, err in zip(insert_idxs, errors):
                if idx >= 0:
                    max_errs[idx] = max(max_errs.get(idx, 0.0), float(err))
                    
            if not max_errs:
                return
                
            updated = False
            # Reverse iterate as sampled games are statistically more likely to be recent
            for meta_i in range(len(self.meta) - 1, -1, -1):
                m = self.meta[meta_i]
                idx = m.get('insert_idx', -1)
                if idx in max_errs:
                    new_q = 0.5 * m.get('quality', 0.5) + 0.5 * min(1.0, max_errs[idx])
                    m['quality'] = new_q
                    # Sync vectorized array
                    if meta_i < self._arr_len:
                        self._quality_arr[meta_i] = new_q
                    updated = True
                    del max_errs[idx]
                    if not max_errs:
                        break
            
            if updated:
                self._weights_dirty = True

    def sample_game_for_reanalyze(self) -> Optional[Tuple[int, 'GameHistory']]:
        """Fetch a random old game from the buffer for MCTS reanalysis.
        Returns: (insert_idx, GameHistory) or None if buffer empty.
        """
        with self._data_lock:
            if not self.meta:
                return None
            
            # Prefer older games for reanalysis (where policy was weak)
            # Sample uniformly from the older 80% of the buffer
            max_idx = max(1, int(len(self.meta) * 0.8))
            meta_idx = np.random.randint(0, max_idx)
            
            game_meta = self.meta[meta_idx]
            cid = game_meta['chunk_id']
            cidx = game_meta['chunk_idx']
            insert_idx = game_meta.get('insert_idx', -1)

            if cid == self.chunk_id_counter: # In active chunk
                if cidx < len(self.active_chunk):
                    return insert_idx, self.active_chunk[cidx]
            else: # On disk
                if cid not in self.chunk_cache:
                    chunk_path = os.path.join(self.chunk_dir, f'chunk_{cid}.pkl')
                    try:
                        with open(chunk_path, 'rb') as f:
                            chunk_games = pickle.load(f)
                        self._add_to_cache(cid, chunk_games)
                    except Exception as e:
                        _log.error("Failed to load chunk %d for reanalyze: %s", cid, e)
                        return None
                if cid in self.chunk_cache and cidx < len(self.chunk_cache[cid]):
                    return insert_idx, self.chunk_cache[cid][cidx]
        return None

    def update_game_by_insert_idx(self, insert_idx: int, new_game: 'GameHistory'):
        """Replace an old game with a reanalyzed version.
        Instead of in-place disk modification, logically deletes the old entry
        and appends the new game to the active chunk.
        """
        if insert_idx < 0:
            return
            
        with self._data_lock:
            old_meta_idx = -1
            for i, m in enumerate(self.meta):
                if m.get('insert_idx') == insert_idx:
                    old_meta_idx = i
                    break
                    
            if old_meta_idx == -1:
                return # Game already evicted during reanalysis

            old_meta = self.meta[old_meta_idx]
            
            # Logically delete old game memory footprint
            self.total_memory -= old_meta['memory_bytes']
            self.meta.pop(old_meta_idx)
        
        # Save as a fresh game
        # Note: training_step sets recency. We use total_games so it's fresh.
        self.save_game(new_game, training_step=self.total_games)


    def _compute_value_target(self, game: GameHistory, pos: int,
                              td_steps: int, discount: float) -> np.ndarray:
        """Compute n-step TD value target (vector) for all players.
        n-step TD: V = sum_i gamma^i r_i + gamma^n V_bootstrap; rotate by player perspective.
        Returns:
            (3,) np.ndarray: [V_me, V_next, V_prev] from perspective of player at `pos`.
        """
        if pos >= len(game):
            # Terminal state
            return np.zeros(3, dtype=np.float32)

        my_player = game.player_ids[pos]
        # Map player ID to index 0,1,2 relative to me: [my, next, prev]
        # Helper to get next/prev player ID
        def next_p(pid): return (pid % 3) + 1
        def prev_p(pid): return ((pid - 2) % 3) + 1
        
        p_ids = [my_player, next_p(my_player), prev_p(my_player)]
        
        value = np.zeros(3, dtype=np.float32)

        for i in range(td_steps):
            idx = pos + i
            if idx < len(game):
                # Intermediate rewards are 0. Only terminal has placement rewards.
                if idx == len(game) - 1 and game.placement_rewards:
                    # Terminal step
                    rews = np.array([game.placement_rewards.get(pid, 0.0) for pid in p_ids], dtype=np.float32)
                    value += (discount ** i) * rews
                    return value # Terminal
            else:
                break

        bootstrap_idx = pos + td_steps
        if bootstrap_idx < len(game):
            # Bootstrap from N-step future root value.
            # root_values[bootstrap_idx] is [V_boot, V_boot+next, V_boot+prev].
            # We want [V_my, V_next, V_prev].
            # Shift = (bootstrap_player_idx - my_player_idx) % 3?
            # bootstrap_idx player is `game.player_ids[bootstrap_idx]`.
            # Let `boot_p` be player at bootstrap_idx.
            # `shift` is how many turns forward `boot_p` is from `my_player`.
            # If boot_p is next(my_player), shift=1. Val[0] is Next. We want it at index 1.
            # Val[2] is Me. We want it at index 0.
            # np.roll(Val, 1) -> [Val[2], Val[0], Val[1]]. Correct.
            # Calculate shift:
            # pid=1 -> 0, pid=2 -> 1, pid=3 -> 2.
            # shift = ( (boot_p - 1) - (my_player - 1) ) % 3
            boot_p = game.player_ids[bootstrap_idx]
            shift = (boot_p - my_player) % 3
            
            bootstrap_val = game.root_values[bootstrap_idx]
            rotated_val = np.roll(bootstrap_val, shift)
            
            value += (discount ** td_steps) * rotated_val

        return value

    # ================================================================
    #  Accessors
    # ================================================================

    def __len__(self):
        return len(self.meta)

    def num_games(self) -> int:
        return len(self.meta)

    def num_positions(self) -> int:
        return sum(m.get('game_len', 0) for m in self.meta)

    def memory_report(self) -> Dict[str, Any]:
        """Return memory usage statistics for logging/monitoring."""
        n_pos = self.num_positions() if len(self.meta) < self._REPORT_NUM_POSITIONS_MAX_GAMES else -1
        return {
            'num_games': len(self.meta),
            'num_positions': n_pos,
            'total_memory_gb': self.total_memory / (1024 ** 3),
            'max_memory_gb': self.max_memory_bytes / (1024 ** 3),
            'usage_pct': 100.0 * self.total_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0,
            'total_evictions': self._eviction_count,
            'avg_quality': float(np.mean([m['quality'] for m in self.meta])) if self.meta else 0.0,
            'avg_game_len': float(np.mean([m.get('game_len', 0) for m in self.meta])) if self.meta else 0.0,
        }

    # ================================================================
    #  Persistence
    # ================================================================

    _SAVE_VERSION = 4

    def save(self, path: str):
        """Save buffer metadata to disk.
        Active chunks are eagerly flushed. Persistent chunks remain in self.chunk_dir.
        """
        try:
            with self._data_lock:
                # Flush the active chunk if not empty
                if len(self.active_chunk) > 0:
                    chunk_id = self.chunk_id_counter
                    chunk_path = os.path.join(self.chunk_dir, f'chunk_{chunk_id}.pkl')
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(self.active_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    self._add_to_cache(chunk_id, self.active_chunk)
                    self.active_chunk = []
                    self.chunk_id_counter += 1

                data = {
                    'version': self._SAVE_VERSION,
                    'is_chunked_v4': True,
                    'total_games': self.total_games,
                    'total_memory': self.total_memory,
                    'max_size': self.max_size,
                    '_eviction_count': self._eviction_count,
                    'chunk_dir': self.chunk_dir,
                    'chunk_id_counter': self.chunk_id_counter,
                    'meta': self.meta
                }
            
            tmp_path = path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception as e:
            _log.error("Failed to save chunked buffer metadata: %s", e)

    def load(self, path: str):
        """Load buffer from disk. Migrates old monolithic buffers into chunks."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            _log.error("Failed to load replay buffer: %s", e)
            return

        version = data.get('version', 1)

        with self._data_lock:
            if data.get('is_chunked_v4'):
                # O(1) Memory Loading! Just metadata.
                self.total_games = data.get('total_games', 0)
                self.max_size = data.get('max_size', self.max_size)
                self._eviction_count = data.get('_eviction_count', 0)
                self.total_memory = data.get('total_memory', 0)
                
                # Keep original chunk_dir if available, else use ours
                loaded_dir = data.get('chunk_dir', self.chunk_dir)
                if os.path.isdir(loaded_dir) and loaded_dir != self.chunk_dir:
                    _log.warning(f"ReplayBuffer loading chunks from different dir: {loaded_dir}")
                    self.chunk_dir = loaded_dir
                    
                self.chunk_id_counter = data.get('chunk_id_counter', 0)
                self.meta = data.get('meta', [])
                self._weights_dirty = True
                self.active_chunk = []
                self.chunk_cache.clear()
                # Rebuild pre-allocated vectorized arrays from loaded meta
                loaded_n = len(self.meta)
                self._arr_capacity = max(loaded_n * 2, 1024)
                self._arr_len = loaded_n
                self._quality_arr = np.empty(self._arr_capacity, dtype=np.float64)
                self._gamelen_arr = np.empty(self._arr_capacity, dtype=np.float64)
                for i, m in enumerate(self.meta):
                    self._quality_arr[i] = m.get('quality', 0.01)
                    self._gamelen_arr[i] = m.get('game_len', 1)
            else:
                _log.warning("Loading legacy monolithic buffer. Initiating format migration...")
                self.active_chunk = []
                self.chunk_cache.clear()
                self.meta = []
                self.total_memory = 0
                self.total_games = 0
                
                # Hack to use old functions safely under lock
                dummy_games = []
                # Reconstruct games list using old methods
                if data.get('is_chunked'):
                    self._load_chunked(data.get('chunk_dir', path + '_chunks'))
                    dummy_games = self.games
                elif version >= 2:
                    self._load_v2(data)
                    dummy_games = self.games
                else:
                    self._load_v1_compat(data)
                    dummy_games = self.games
                
                self.games = [] # Delete reference immediately
                self.meta = []
                self.total_memory = 0
                self.total_games = 0
                self.chunk_id_counter = 0
                
                # Migrate games into chunked format effortlessly
                print(f"[ReplayBuffer] Migrating {len(dummy_games)} games to Chunked Ring Buffer...", flush=True)
                for g in dummy_games:
                    if getattr(g, 'board_states', None) is None:
                        self._precompute_focus_data(g)
                    self._save_game_locked(g, training_step=0)
                
                # Free memory
                del dummy_games
                print("[ReplayBuffer] Migration complete.", flush=True)

        # Enforce budget post-load
        with self._data_lock:
            self._maybe_evict()

        report = self.memory_report()
        print(f"[ReplayBuffer] Active {report['num_games']} games "
              f"({report['total_memory_gb']:.2f} GB / {report['max_memory_gb']:.1f} GB, "
              f"{report['usage_pct']:.1f}%, avg_q={report['avg_quality']:.3f})",
              flush=True)

    def _load_chunked(self, chunk_dir: str):
        """Load from a chunked directory format."""
        meta_path = os.path.join(chunk_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            _log.error("Chunk meta.pkl not found: %s", meta_path)
            return
            
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            
        self.games = []
        self.meta = []
        skipped = 0
        
        for i in range(data.get('num_chunks', 0)):
            c_path = os.path.join(chunk_dir, f'chunk_{i}.pkl')
            if not os.path.exists(c_path):
                _log.warning("Missing chunk %d", i)
                continue
            try:
                with open(c_path, 'rb') as f:
                    c_data = pickle.load(f)
                for g in c_data.get('games', []):
                    if not self._is_valid_game_entry(g):
                        skipped += 1
                        continue
                    self.games.append(g)
                for m in c_data.get('meta', []):
                    self.meta.append(m)
            except Exception as e:
                 _log.error("Failed to load chunk %d: %s", i, e)

        if skipped > 0:
            print(f"[ReplayBuffer] Skipped {skipped} corrupted game entries.", flush=True)

        self.total_games = data.get('total_games', len(self.games))
        self.max_size = data.get('max_size', self.max_size)
        self._eviction_count = data.get('_eviction_count', 0)
        self._weights_dirty = True
        
        while len(self.meta) < len(self.games):
            idx = len(self.meta)
            self.meta.append({
                'insert_idx': self.total_games - len(self.games) + idx,
                'training_step': 0,
                'memory_bytes': self._estimate_game_memory(self.games[idx]) if idx < len(self.games) else 0,
                'quality': 0.5,
                'timestamp': time.time()
            })

    def _load_v2(self, data: Dict):
        """Load new format (version 2+). Only append entries that look like valid GameHistory."""
        raw_games = data.get('games', []) if isinstance(data.get('games'), list) else []
        raw_meta = data.get('meta', []) if isinstance(data.get('meta'), list) else []
        skipped = 0
        self.games = []
        self.meta = []

        for i, g in enumerate(raw_games):
            try:
                if not self._is_valid_game_entry(g):
                    skipped += 1
                    continue
                self.games.append(g)
                if i < len(raw_meta) and isinstance(raw_meta[i], dict):
                    self.meta.append(raw_meta[i])
                else:
                    self.meta.append({
                        'insert_idx': len(self.games) - 1,
                        'training_step': 0,
                        'memory_bytes': self._estimate_game_memory(g),
                        'quality': 0.5,
                        'timestamp': time.time(),
                    })
            except Exception:
                skipped += 1

        if skipped > 0:
            print(f"[ReplayBuffer] Skipped {skipped} invalid or corrupted game entries.", flush=True)

        self.total_games = data.get('total_games', len(self.games))
        self.max_size = data.get('max_size', self.max_size)
        self._eviction_count = data.get('_eviction_count', 0)
        self._weights_dirty = True

        # Ensure meta length matches games
        while len(self.meta) < len(self.games):
            idx = len(self.meta)
            self.meta.append({
                'insert_idx': self.total_games - len(self.games) + idx,
                'training_step': 0,
                'memory_bytes': self._estimate_game_memory(self.games[idx]) if idx < len(self.games) else 0,
                'quality': 0.5,
                'timestamp': time.time(),
            })

    def _load_v1_compat(self, data: Dict):
        """Load old format (deque-based, version 1)."""
        old_buffer = data.get('buffer', [])
        self.total_games = data.get('total_games', len(old_buffer))
        self.max_size = data.get('max_size', self.max_size)
        self._eviction_count = 0
        self._weights_dirty = True

        print(f"[ReplayBuffer] Converting v1 format ({len(old_buffer)} games)...", flush=True)

        self.games = []
        self.meta = []
        for i, game in enumerate(old_buffer):
            mem = self._estimate_game_memory(game)
            meta = {
                'insert_idx': i,
                'training_step': 0,
                'memory_bytes': mem,
                'quality': 0.5,  # Will be recomputed
                'timestamp': time.time(),
            }
            self.games.append(game)
            self.meta.append(meta)

        # Compute quality scores for all loaded games
        for i in range(len(self.games)):
            # Upgrade legacy scalar root_values to vector
            if len(self.games[i].root_values) > 0:
                first_val = self.games[i].root_values[0]
                if isinstance(first_val, (float, int, np.floating, np.integer)):
                    # Convert all scalars to [v, 0, 0]
                    # Note: scalar v was "current player win rate".
                    # [v, (1-v)/2, (1-v)/2] might be better but 0 is safe for now as 
                    # we rely on placement rewards for terminal anyway.
                    new_vals = []
                    for v in self.games[i].root_values:
                        vec = np.zeros(3, dtype=np.float32)
                        vec[0] = float(v)
                        new_vals.append(vec)
                    self.games[i].root_values = new_vals
                    
            self.meta[i]['quality'] = self._compute_quality(self.games[i], self.meta[i])

    def _is_valid_game_entry(self, g: Any) -> bool:
        """Return True if g looks like a valid GameHistory (has required list attrs)."""
        if g is None:
            return False
        for attr in ('observations', 'actions', 'rewards', 'policy_targets', 'root_values'):
            val = getattr(g, attr, None)
            if not isinstance(val, list):
                return False
        return True

    def _ensure_precomputed(self):
        """Backward compat: precompute focus data for old games missing it. Skip bad entries per-game."""
        count = 0
        skip_count = 0
        for game in self.games:
            try:
                if not self._is_valid_game_entry(game):
                    skip_count += 1
                    continue
                if getattr(game, 'board_states', None) is None:
                    self._precompute_focus_data(game)
                    count += 1
            except Exception:
                skip_count += 1
        if skip_count > 0:
            print(f"[ReplayBuffer] Skipped precompute for {skip_count} invalid game(s).", flush=True)
        if count > 0:
            print(f"[ReplayBuffer] Precomputed focus data for {count} legacy games.", flush=True)

    def _filter_corrupted_games(self):
        """Remove games with NaN/Inf data. Per-game try/except so one bad game doesn't kill the run."""
        valid_games = []
        valid_meta = []
        corrupted = 0
        meta_list = list(self.meta)
        while len(meta_list) < len(self.games):
            meta_list.append({'insert_idx': len(meta_list), 'training_step': 0, 'memory_bytes': 0, 'quality': 0.5, 'timestamp': time.time()})

        for game, meta in zip(self.games, meta_list):
            is_valid = True
            try:
                # Check observations
                obs_list = getattr(game, 'observations', None)
                if obs_list is None or not isinstance(obs_list, list):
                    is_valid = False
                else:
                    for obs in obs_list:
                        if np.isnan(obs).any() or np.isinf(obs).any():
                            is_valid = False
                            break

                # Check rewards
                if is_valid:
                    rewards_list = getattr(game, 'rewards', None)
                    if rewards_list is None or not isinstance(rewards_list, list):
                        is_valid = False
                    else:
                        for reward in rewards_list:
                            if np.isnan(reward) or np.isinf(reward):
                                is_valid = False
                                break

                # Check root values
                if is_valid:
                    rv_list = getattr(game, 'root_values', None)
                    if rv_list is None or not isinstance(rv_list, list):
                        is_valid = False
                    else:
                        for val in rv_list:
                            v = np.asarray(val)
                            if np.isnan(v).any() or np.isinf(v).any():
                                is_valid = False
                                break

                # Check policy targets
                if is_valid:
                    pt_list = getattr(game, 'policy_targets', None)
                    if pt_list is None or not isinstance(pt_list, list):
                        is_valid = False
                    else:
                        for p in pt_list:
                            if np.isnan(p).any() or np.isinf(p).any():
                                is_valid = False
                                break
            except Exception:
                is_valid = False

            if is_valid:
                valid_games.append(game)
                valid_meta.append(meta)
            else:
                corrupted += 1

        if corrupted > 0:
            _log.warning("Discarded %d corrupted games (NaN/Inf or invalid).", corrupted)

        self.games = valid_games
        self.meta = valid_meta
        self._weights_dirty = True

    def _recompute_memory(self):
        """Recompute total_memory from game-level estimates."""
        self.total_memory = 0
        for i, game in enumerate(self.games):
            mem = self._estimate_game_memory(game)
            self.meta[i]['memory_bytes'] = mem
            self.total_memory += mem

    def _enforce_memory_budget(self):
        """After loading, evict if over memory/count budget."""
        if self.total_memory > self.max_memory_bytes or len(self.games) > self.max_size:
            over_mem = self.total_memory - self.max_memory_bytes
            over_cnt = len(self.games) - self.max_size
            _log.warning(
                "Loaded buffer exceeds budget (mem: +%.0f MB, count: +%d). Running eviction...",
                max(0, over_mem) / (1024**2), max(0, over_cnt)
            )
            target_mem = int(self.max_memory_bytes * self.EVICTION_TARGET_RATIO)
            target_cnt = int(self.max_size * 0.90)
            reason = "memory" if self.total_memory > self.max_memory_bytes else "count"
            self._run_eviction(target_mem, target_cnt, reason=reason)
