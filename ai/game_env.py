"""Game environment for Eight-in-a-Row (MuZero integration).

Maintainability: see ai/MAINTENANCE.md."""

import numpy as np
from typing import Optional, Tuple, List, Dict

from ai.log_utils import get_logger

_log = get_logger(__name__)


class EightInARowEnv:
    """Three-player N-in-a-Row game environment (Variable Size)."""

    # Defaults (overridden by __init__)
    DEFAULT_BOARD_SIZE = 100
    DEFAULT_WIN_LENGTH = 8
    NUM_PLAYERS = 3
    PLAYER_IDS = [1, 2, 3]  # 1=red, 2=green, 3=blue

    # Placement rewards: 5:2:0 points mapped to [-1, +1]
    PLACEMENT_POINTS = (5, 2, 0)
    PLACEMENT_REWARDS = (1.0, -0.2, -1.0)

    def __init__(self, board_size: int = 100, win_length: int = 8):
        board_size = int(board_size)
        win_length = int(win_length)
        if board_size < 3 or board_size > 200:
            raise ValueError(
                f"board_size must be in [3, 200], got {board_size}. "
                "Use a positive integer within this range."
            )
        if win_length < 2 or win_length > board_size:
            raise ValueError(
                f"win_length must be in [2, board_size] (board_size={board_size}), got {win_length}."
            )
        self.BOARD_SIZE = board_size
        self.WIN_LENGTH = win_length
        
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = 0  # index into PLAYER_IDS
        self.move_history: List[Tuple[int, int, int]] = []  # (row, col, player_id)
        self.done = False
        self.winner: Optional[int] = None  # player_id or None
        # Full rankings: [(pid, placement_0indexed)] sorted 1st→3rd
        self.rankings: List[Tuple[int, int]] = []
        # Per-player placement reward: {pid: reward}
        self.placement_rewards: Dict[int, float] = {}
        # Track bounding box of placed pieces for local view
        self._min_r = self.BOARD_SIZE
        self._max_r = -1
        self._min_c = self.BOARD_SIZE
        self._max_c = -1
        # Cached per-player board planes (updated incrementally in step())
        self._player_planes = np.zeros((4, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        self._player_planes[3] = 1.0  # empty channel starts all-ones
        self._planes_dirty = True  # True when current_player changed (rotation needed)
        # Rotation cache: avoid recomputing _get_rotated_planes when player hasn't changed
        self._cached_rotated: Optional[np.ndarray] = None
        self._cached_rotated_player: int = -1
        # Precomputed pooling indices for pure-numpy area interpolation (size -> 21)
        self._pool_rows = np.linspace(0, self.BOARD_SIZE, 22, dtype=int)  # 22 boundaries → 21 bins
        self._pool_cols = np.linspace(0, self.BOARD_SIZE, 22, dtype=int)
        # Precompute bin areas for normalization
        dr = np.diff(self._pool_rows)  # (21,)
        dc = np.diff(self._pool_cols)  # (21,)
        self._pool_areas = (dr[:, None] * dc[None, :]).astype(np.float32)  # (21, 21)

    def reset(self):
        """Reset the environment to initial state."""
        self.board.fill(0)
        self.current_player = 0
        self.move_history.clear()
        self.done = False
        self.winner = None
        self.rankings = []
        self.placement_rewards = {}
        self._min_r = self.BOARD_SIZE
        self._max_r = -1
        self._min_c = self.BOARD_SIZE
        self._max_c = -1
        self._player_planes = np.zeros((4, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        self._player_planes[3] = 1.0
        self._planes_dirty = True
        self._cached_rotated = None
        self._cached_rotated_player = -1
        return self

    def clone(self) -> 'EightInARowEnv':
        """Create a deep copy of this environment."""
        env = EightInARowEnv(board_size=self.BOARD_SIZE, win_length=self.WIN_LENGTH)
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.move_history = list(self.move_history)
        env.done = self.done
        env.winner = self.winner
        env.rankings = list(self.rankings)
        env.placement_rewards = dict(self.placement_rewards)
        env._min_r = self._min_r
        env._max_r = self._max_r
        env._min_c = self._min_c
        env._max_c = self._max_c
        env._player_planes = self._player_planes.copy()
        env._planes_dirty = self._planes_dirty
        return env

    @property
    def current_player_id(self) -> int:
        return self.PLAYER_IDS[self.current_player]

    def get_center(self) -> Tuple[int, int]:
        """Get the geometric center of the active region."""
        if self._max_r < 0:
            return self.BOARD_SIZE // 2, self.BOARD_SIZE // 2
        cr = (self._min_r + self._max_r) // 2
        cc = (self._min_c + self._max_c) // 2
        return cr, cc

    def get_smart_center(self, view_size: int = 21) -> Tuple[int, int]:
        """
        Get a center that ensures the local view has valid moves.
        If the geometric center is clogged (full), shift to the periphery
        of the stone cluster to find empty spots.
        """
        # 1. Try standard geometric center
        cr, cc = self.get_center()
        
        # Helper to check if a center is valid (has legal moves)
        def is_valid_center(r, c):
            moves = self.legal_moves_local(r, c, view_size)
            return len(moves) > 0

        # Optimization: if board is empty or center is valid, return it
        if self._max_r < 0 or is_valid_center(cr, cc):
            return cr, cc

        # 2. Geometric center is clogged. Try candidates on the periphery.
        # We look at corners and midpoints of the bounding box.
        candidates = []
        
        # Bounding box
        min_r, max_r = self._min_r, self._max_r
        min_c, max_c = self._min_c, self._max_c
        
        # Corners
        candidates.append((min_r, min_c))
        candidates.append((min_r, max_c))
        candidates.append((max_r, min_c))
        candidates.append((max_r, max_c))
        
        # Midpoints
        mid_r = (min_r + max_r) // 2
        mid_c = (min_c + max_c) // 2
        candidates.append((min_r, mid_c))
        candidates.append((max_r, mid_c))
        candidates.append((mid_r, min_c))
        candidates.append((mid_r, max_c))

        # Check candidates
        for r, c in candidates:
            # We must clamp them to ensure the view stays (mostly) on board? 
            # legal_moves_local handles board boundaries, but let's ensure center is essentially sane.
            # Actually, legal_moves_local handles clamped loops.
            if is_valid_center(r, c):
                return r, c

        # 3. If all fail, return original center (truly full or weird edge case)
        return cr, cc

    def step(self, row: int, col: int) -> Tuple[float, bool]:
        """Place a piece. Returns (reward_for_current_player, done)."""
        if self.done:
            raise RuntimeError("Game already over")
        if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
            raise ValueError(f"Invalid position ({row},{col}), must be in [0,{self.BOARD_SIZE})")
        if self.board[row, col] != 0:
            raise ValueError(f"Cell ({row},{col}) already occupied by player {self.board[row, col]}")

        pid = self.current_player_id
        self.board[row, col] = pid
        self.move_history.append((row, col, pid))

        # Update bounding box
        self._min_r = min(self._min_r, row)
        self._max_r = max(self._max_r, row)
        self._min_c = min(self._min_c, col)
        self._max_c = max(self._max_c, col)

        # Incrementally update raw per-player planes (absolute, not rotated)
        # Channel idx = pid - 1 for players 1,2,3; channel 3 = empty
        self._player_planes[pid - 1, row, col] = 1.0
        self._player_planes[3, row, col] = 0.0
        self._planes_dirty = True
        self._cached_rotated = None  # invalidate rotation cache

        # Check win
        max_chain = self._get_max_chain_length(row, col, pid)
        if max_chain >= self.WIN_LENGTH:
            self.done = True
            self.winner = pid
            self.rank_players()
            reward = 1.0
        elif not np.any(self._player_planes[3]):
            # Board full — draw (no empty cells remain)
            self.done = True
            self.winner = None
            self._rank_draw()
            reward = 0.0
        else:
            reward = 0.0
            
            # --- Reward Shaping (Dense Aux Reward) ---
            # 1. Did we create a WIN_LENGTH - 1 chain? (e.g., 4-in-a-row)
            if max_chain == self.WIN_LENGTH - 1:
                reward = 0.05
            else:
                # 2. Did we block an opponent's WIN_LENGTH - 1 chain?
                blocked_threat = False
                for opp_idx in range(1, self.NUM_PLAYERS):
                    opp_pid = self.PLAYER_IDS[(self.current_player + opp_idx) % self.NUM_PLAYERS]
                    self.board[row, col] = opp_pid
                    opp_chain = self._get_max_chain_length(row, col, opp_pid)
                    if opp_chain >= self.WIN_LENGTH - 1:
                        blocked_threat = True
                self.board[row, col] = pid # Restore state
                
                if blocked_threat:
                    reward = 0.05

            self.current_player = (self.current_player + 1) % self.NUM_PLAYERS

        return reward, self.done

    def _get_max_chain_length(self, row: int, col: int, pid: int) -> int:
        """Get the maximum chain length created by placing a piece at (row,col)."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_chain = 1
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                for i in range(1, self.BOARD_SIZE):
                    nr, nc = row + dr * i * sign, col + dc * i * sign
                    if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                        if self.board[nr, nc] == pid:
                            count += 1
                        else:
                            break
                    else:
                        break
            if count > max_chain:
                max_chain = count
        return max_chain

    def _check_win(self, row: int, col: int, pid: int) -> bool:
        """Check if placing at (row,col) creates WIN_LENGTH in a row."""
        return self._get_max_chain_length(row, col, pid) >= self.WIN_LENGTH

    # ================================================================
    #  Chain Analysis & Ranking
    # ================================================================
    def get_threat_levels(self) -> np.ndarray:
        """
        Check for existence of chains of length 5, 6, and 7+ for any player.
        Returns:
            np.array([has_5, has_6, has_7], dtype=np.float32)
            where 1.0 means at least one chain of that length exists, 0.0 otherwise.
        """
        threats = np.zeros(3, dtype=np.float32) # [5, 6, 7+]
        
        # We need to check all players. 
        # Since compute_chain_profile is fast enough (O(N^2)), we can just run it for all active players.
        # Check active players (1, 2, 3)
        for pid in self.PLAYER_IDS:
            profile = self.compute_chain_profile(pid)
            # profile is {length: count}
            
            # Check length 5
            if profile.get(5, 0) > 0:
                threats[0] = 1.0
            
            # Check length 6
            if profile.get(6, 0) > 0:
                threats[1] = 1.0
                
            # Check length 7 (and higher? User said 5/6/7. 8 is win/terminal)
            # If 8 exists, game is over, usually.
            # We'll check 7 explicitly.
            if profile.get(7, 0) > 0:
                threats[2] = 1.0
                
            # Early exit if all found?
            if threats.all():
                break
                
        return threats

    @staticmethod
    def _rle_profile(mask_1d: np.ndarray) -> Dict[int, int]:
        """Run-length encode a 1-D boolean array and return {run_length: count}.

        Uses np.diff to find run boundaries in O(1) Python calls.
        """
        if mask_1d.size == 0:
            return {}
        # Pad with False at both ends so diff always catches start/end of runs
        padded = np.empty(mask_1d.size + 2, dtype=np.bool_)
        padded[0] = False
        padded[-1] = False
        padded[1:-1] = mask_1d
        d = np.diff(padded.view(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if starts.size == 0:
            return {}
        lengths = ends - starts
        counts = np.bincount(lengths)
        return {int(l): int(counts[l]) for l in range(1, counts.size) if counts[l] > 0}

    def compute_chain_profile(self, pid: int) -> Dict[int, int]:
        """Count all consecutive chains for a player in all 4 directions.

        Returns a dict {chain_length: count}, e.g. {7: 5, 6: 3, 5: 10}.
        Only chains of length >= 1 are included.

        Vectorized: uses NumPy run-length encoding instead of Python loops.
        """
        board = self.board
        size = self.BOARD_SIZE
        mask = (board == pid)  # (size, size) bool

        profile: Dict[int, int] = {}

        def _merge(sub: Dict[int, int]):
            for k, v in sub.items():
                profile[k] = profile.get(k, 0) + v

        # 1. Horizontal: each row is a 1-D sequence
        for r in range(size):
            _merge(self._rle_profile(mask[r]))

        # 2. Vertical: each column is a 1-D sequence
        for c in range(size):
            _merge(self._rle_profile(mask[:, c]))

        # 3. Diagonal ↘ (offset from -size+1 to size-1)
        for offset in range(-size + 1, size):
            diag = np.diag(mask, offset)
            _merge(self._rle_profile(diag))

        # 4. Anti-diagonal ↙ (flip columns, then take diagonals)
        flipped = mask[:, ::-1]
        for offset in range(-size + 1, size):
            diag = np.diag(flipped, offset)
            _merge(self._rle_profile(diag))

        return profile

    def _chain_sort_key(self, pid: int) -> List[Tuple[int, int]]:
        """Generate a sort key from chain profile for ranking comparison.

        Returns a descending-sorted list of (chain_length, count) tuples.
        Python's natural tuple/list comparison gives correct ordering:
        longer chains first, then more chains of same length.
        """
        profile = self.compute_chain_profile(pid)
        return sorted(profile.items(), key=lambda x: x[0], reverse=True)

    def rank_players(self):
        """Rank all players after game ends.

        Winner is 1st place. Remaining players ranked by chain profile:
        the player with a better chain profile (longest chain > more groups
        > second-longest chain > ...) gets 2nd place.

        Sets self.rankings and self.placement_rewards.
        """
        if not self.done or self.winner is None:
            return

        winner = self.winner
        losers = [pid for pid in self.PLAYER_IDS if pid != winner]

        # Sort losers by chain profile descending (better profile = higher rank)
        loser_keys = [(pid, self._chain_sort_key(pid)) for pid in losers]
        loser_keys.sort(key=lambda x: x[1], reverse=True)

        # Build rankings: [(pid, placement_0indexed)]
        self.rankings = [(winner, 0)]  # 1st place
        for i, (pid, _) in enumerate(loser_keys):
            self.rankings.append((pid, i + 1))  # 2nd, 3rd place

        # Build placement rewards: {pid: reward}
        self.placement_rewards = {}
        for pid, placement in self.rankings:
            self.placement_rewards[pid] = self.PLACEMENT_REWARDS[placement]

    def _rank_draw(self):
        """Rank all players when the game ends in a draw (board full, no winner).

        All players are ranked by chain profile (best chains = higher rank).
        Uses the same placement rewards as normal games.
        """
        if not self.done:
            return

        # Sort all players by chain profile descending
        player_keys = [(pid, self._chain_sort_key(pid)) for pid in self.PLAYER_IDS]
        player_keys.sort(key=lambda x: x[1], reverse=True)

        self.rankings = [(pid, i) for i, (pid, _) in enumerate(player_keys)]

        self.placement_rewards = {}
        for pid, placement in self.rankings:
            self.placement_rewards[pid] = self.PLACEMENT_REWARDS[placement]

    def legal_moves(self) -> np.ndarray:
        """Return array of (row, col) pairs for all empty cells."""
        return np.argwhere(self.board == 0)

    def legal_moves_local(self, center_r: int, center_c: int, view_size: int = 21) -> List[Tuple[int, int]]:
        """Return legal moves within the local view window."""
        half = view_size // 2
        r_start = max(0, center_r - half)
        r_end = min(self.BOARD_SIZE, center_r + half + 1)
        c_start = max(0, center_c - half)
        c_end = min(self.BOARD_SIZE, center_c + half + 1)
        local = self.board[r_start:r_end, c_start:c_end]
        coords = np.argwhere(local == 0)
        return [(int(r_start + r), int(c_start + c)) for r, c in coords]

    def _get_rotated_planes(self) -> np.ndarray:
        """Return (4, 100, 100) planes rotated to current player's perspective.

        Uses cached per-player planes (updated incrementally in step()) and
        rotates [current, next, prev, empty] based on current_player.
        """
        pid = self.current_player_id
        next_pid = self.PLAYER_IDS[(self.current_player + 1) % self.NUM_PLAYERS]
        prev_pid = self.PLAYER_IDS[(self.current_player + 2) % self.NUM_PLAYERS]
        out = np.empty((4, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        out[0] = self._player_planes[pid - 1]
        out[1] = self._player_planes[next_pid - 1]
        out[2] = self._player_planes[prev_pid - 1]
        out[3] = self._player_planes[3]
        return out

    def _get_rotated_planes_cached(self) -> np.ndarray:
        """Cached version of _get_rotated_planes.

        Returns the same result within a single move (between step() calls).
        Cache is invalidated in step() when a piece is placed.
        """
        if self._cached_rotated is not None and self._cached_rotated_player == self.current_player:
            return self._cached_rotated
        self._cached_rotated = self._get_rotated_planes()
        self._cached_rotated_player = self.current_player
        return self._cached_rotated

    def _numpy_area_pool(self, planes: np.ndarray, target_size: int) -> np.ndarray:
        """Pure-numpy area interpolation."""
        if self.BOARD_SIZE < target_size:
            _, h, w = planes.shape
            pad_h = target_size - h
            pad_w = target_size - w
            return np.pad(planes, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # Lazy init for pooling arrays based on target_size
        if not hasattr(self, '_pool_cache') or self._pool_cache.get('size') != target_size:
            pr = np.linspace(0, self.BOARD_SIZE, target_size + 1, dtype=int)
            pc = np.linspace(0, self.BOARD_SIZE, target_size + 1, dtype=int)
            pa = (np.diff(pr)[:, None] * np.diff(pc)[None, :]).astype(np.float32)
            self._pool_cache = {'size': target_size, 'rows': pr, 'cols': pc, 'areas': pa}
            
        pr = self._pool_cache['rows']
        pc = self._pool_cache['cols']
        pa = self._pool_cache['areas']

        row_sums = np.add.reduceat(planes, pr[:-1], axis=1)
        bin_sums = np.add.reduceat(row_sums, pc[:-1], axis=2)
        return bin_sums / pa

    def get_global_state(self) -> np.ndarray:
        """
        Get global state for Focus Network: (1, 4, 100, 100).
        Channels: [current, next, prev, empty]
        Uses cached rotated planes for speed.
        """
        return self._get_rotated_planes_cached()[np.newaxis, ...]

    def get_observation(self, view_size: int = 21, center: Optional[Tuple[int, int]] = None,
                        rotated_planes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Get observation: 8 channels (4 Local + 4 Global Thumbnail).
        
        Channels 0-3: Local view (21x21) - [current, next, prev, empty]
        Channels 4-7: Global view (100x100 -> 21x21 heatmaps) - [current, next, prev, empty]

        Args:
            rotated_planes: pre-computed rotated planes to avoid recomputation.
                            If None, computes internally (cached).
        """
        # --- Local View ---
        if center is not None:
            cr, cc = center
        else:
            cr, cc = self.get_smart_center(view_size)

        half = view_size // 2
        cr = max(half, min(self.BOARD_SIZE - half - 1, cr))
        cc = max(half, min(self.BOARD_SIZE - half - 1, cc))

        r_start = cr - half
        c_start = cc - half

        # Reuse provided planes or get cached
        rotated = rotated_planes if rotated_planes is not None else self._get_rotated_planes_cached()

        # Local view = crop of rotated planes
        obs_local = rotated[:, r_start:r_start + view_size, c_start:c_start + view_size].copy()

        # Phase 3 Fix: Pad if smaller than view_size (e.g. 15x15 board)
        _, h, w = obs_local.shape
        if h < view_size or w < view_size:
            # Since r_start >= 0 due to clamping, truncation only happens at bottom/right
            pad_h = view_size - h
            pad_w = view_size - w
            obs_local = np.pad(obs_local, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # --- Global Thumbnail (pure numpy, no torch) ---
        obs_global = self._numpy_area_pool(rotated, view_size)

        # --- Concatenate ---
        obs = np.concatenate([obs_local, obs_global], axis=0)  # (8, view_size, view_size)

        return obs, (cr, cc)

    def action_to_board(self, action: int, center_r: int, center_c: int,
                        view_size: int = 21) -> Tuple[int, int]:
        """Convert local action index to board (row, col). Validates action range."""
        policy_size = view_size * view_size
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= policy_size:
            raise ValueError(
                f"action must be in [0, {policy_size}), got {action!r}"
            )
        half = view_size // 2
        local_r = action // view_size
        local_c = action % view_size
        board_r = center_r - half + local_r
        board_c = center_c - half + local_c
        # Clamp to board bounds so callers never get out-of-range indices
        board_r = max(0, min(self.BOARD_SIZE - 1, board_r))
        board_c = max(0, min(self.BOARD_SIZE - 1, board_c))
        return board_r, board_c

    def board_to_action(self, row: int, col: int, center_r: int, center_c: int,
                        view_size: int = 21) -> int:
        """Convert board (row, col) to local action index."""
        half = view_size // 2
        local_r = row - (center_r - half)
        local_c = col - (center_c - half)
        return local_r * view_size + local_c

    def get_legal_actions_mask(self, center_r: int, center_c: int,
                                view_size: int = 21) -> np.ndarray:
        """Return a flat boolean mask for legal actions in the local view (vectorized)."""
        half = view_size // 2
        r_start = center_r - half
        c_start = center_c - half
        r_end = r_start + view_size
        c_end = c_start + view_size

        # Clamp board extraction bounds
        br_start = max(0, r_start)
        br_end = min(self.BOARD_SIZE, r_end)
        bc_start = max(0, c_start)
        bc_end = min(self.BOARD_SIZE, c_end)

        # Extract local board region and check empty cells
        local = self.board[br_start:br_end, bc_start:bc_end]
        local_mask = (local == 0).astype(np.float32)

        # Place into the full view_size x view_size mask (handles edge padding)
        mask = np.zeros((view_size, view_size), dtype=np.float32)
        lr_off = br_start - r_start
        lc_off = bc_start - c_start
        mask[lr_off:lr_off + local_mask.shape[0],
             lc_off:lc_off + local_mask.shape[1]] = local_mask

        return mask.ravel()

    def get_legal_moves_and_mask(self, center_r: int, center_c: int,
                                  view_size: int = 21) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Return both legal moves list and flat mask in one pass (avoids double scan)."""
        half = view_size // 2
        r_start = center_r - half
        c_start = center_c - half
        r_end = r_start + view_size
        c_end = c_start + view_size

        br_start = max(0, r_start)
        br_end = min(self.BOARD_SIZE, r_end)
        bc_start = max(0, c_start)
        bc_end = min(self.BOARD_SIZE, c_end)

        local = self.board[br_start:br_end, bc_start:bc_end]
        local_mask = (local == 0).astype(np.float32)

        mask = np.zeros((view_size, view_size), dtype=np.float32)
        lr_off = br_start - r_start
        lc_off = bc_start - c_start
        mask[lr_off:lr_off + local_mask.shape[0],
             lc_off:lc_off + local_mask.shape[1]] = local_mask

        coords = np.argwhere(local == 0)
        moves = [(int(br_start + r), int(bc_start + c)) for r, c in coords]

        return moves, mask.ravel()

    def get_reward(self, player_id: int) -> float:
        """Get terminal reward for a specific player using placement-based rewards.

        Returns: +1.0 (1st), -0.2 (2nd), -1.0 (3rd), or 0.0 if game not done.
        """
        if not self.done:
            return 0.0
        # Use placement rewards if rankings have been computed
        if self.placement_rewards:
            return self.placement_rewards.get(player_id, -1.0)
        # Fallback for backward compatibility
        if self.winner == player_id:
            return 1.0
        return -1.0

    def render_ascii(self, view_size: int = 21) -> str:
        """Render the active region as ASCII for debugging."""
        cr, cc = self.get_center()
        half = view_size // 2
        cr = max(half, min(self.BOARD_SIZE - half - 1, cr))
        cc = max(half, min(self.BOARD_SIZE - half - 1, cc))
        symbols = {0: '.', 1: 'R', 2: 'G', 3: 'B'}
        lines = []
        for r in range(cr - half, cr + half + 1):
            row_str = ''
            for c in range(cc - half, cc + half + 1):
                row_str += symbols.get(self.board[r, c], '?') + ' '
            lines.append(row_str)
        return '\n'.join(lines)
