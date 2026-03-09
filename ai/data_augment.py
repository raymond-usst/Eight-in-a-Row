"""Board data augmentation for more training diversity.

Available methods:
1. Rotation + mirror (default): 0/90/180/270° and horizontal flip → 8 symmetries.
2. Observation noise (optional): additive Gaussian noise on observations for robustness.

Not used here:
- Player/color permutation: state is already from current player's perspective (me/next/prev),
  so permuting 1/2/3 does not change the batch.
- Vertical mirror: already covered by rotation + horizontal mirror.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from ai.log_utils import get_logger

_log = get_logger(__name__)

VIEW_SIZE = 21

# ================================================================
#  Permutation cache (precomputed at module load — avoids per-sample 441-loop)
# ================================================================
def _rot90_ccw(r: int, c: int, k: int, size: int) -> Tuple[int, int]:
    """Apply k*90° counter-clockwise rotation. (r,c) in [0, size-1]."""
    if k == 0:
        return r, c
    if k == 1:
        return c, size - 1 - r
    if k == 2:
        return size - 1 - r, size - 1 - c
    if k == 3:
        return size - 1 - c, r
    return r, c


def _inv_rot90_ccw(r: int, c: int, k: int, size: int) -> Tuple[int, int]:
    """Inverse of _rot90_ccw (so we map new view back to old view)."""
    if k == 0:
        return r, c
    if k == 1:
        return size - 1 - c, r
    if k == 2:
        return size - 1 - r, size - 1 - c
    if k == 3:
        return c, size - 1 - r
    return r, c


def _build_action_perms(view_size: int) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Precompute (inv_perm, fwd_perm) for all 8 (rot, mirror) combinations."""
    action_size = view_size * view_size
    cache = {}
    for rot in range(4):
        for mirror in range(2):
            inv_perm = np.zeros(action_size, dtype=np.int32)
            fwd_perm = np.zeros(action_size, dtype=np.int32)
            for new_r in range(view_size):
                for new_c in range(view_size):
                    r1, c1 = new_r, view_size - 1 - new_c if mirror else new_c
                    old_r, old_c = _inv_rot90_ccw(r1, c1, rot, view_size)
                    old_idx = old_r * view_size + old_c
                    new_idx = new_r * view_size + new_c
                    inv_perm[new_idx] = old_idx
                    fwd_perm[old_idx] = new_idx
            cache[(rot, mirror)] = (inv_perm, fwd_perm)
    return cache


# Module-level cache: computed once at import time (~0.5ms)
_PERM_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = _build_action_perms(VIEW_SIZE)


def _action_inv_perm_21(rot: int, mirror: int) -> np.ndarray:
    """Cached inverse permutation lookup."""
    return _PERM_CACHE[(rot, mirror)][0]


def _action_fwd_perm_21(rot: int, mirror: int) -> np.ndarray:
    """Cached forward permutation lookup."""
    return _PERM_CACHE[(rot, mirror)][1]


def _center_transform(row: int, col: int, board_size: int, rot: int, mirror: int) -> Tuple[int, int]:
    """Transform board (row, col) by rot then mirror. Returns (new_row, new_col)."""
    r, c = _rot90_ccw(row, col, rot, board_size)
    if mirror:
        c = board_size - 1 - c
    return r, c


def _transform_spatial_nd(x: np.ndarray, rot: int, mirror: int) -> np.ndarray:
    """Apply rot (0..3) then horizontal flip to last two dimensions. x: (..., H, W)."""
    if rot != 0:
        x = np.rot90(x, rot, axes=(-2, -1)).copy()
    if mirror:
        x = np.flip(x, axis=-1).copy()
    return x


def apply_observation_noise(
    batch: Dict[str, np.ndarray], rng: np.random.Generator, std: float = 0.02
) -> None:
    """Add Gaussian noise to observations and next_observations (vectorized). In-place."""
    if std <= 0:
        return
    if batch is None or not isinstance(batch, dict):
        raise ValueError("batch must be a non-empty dict")
    obs = batch.get('observations')
    if obs is None or not isinstance(obs, np.ndarray) or obs.size == 0:
        raise ValueError("batch['observations'] must be a non-empty numpy array")
    
    # Vectorized: generate noise for entire batch at once
    noise = rng.normal(0, std, obs.shape).astype(np.float32)
    np.clip(obs + noise, 0.0, 1.0, out=batch['observations'])
    
    next_obs = batch.get('next_observations')
    if next_obs is not None and isinstance(next_obs, np.ndarray) and next_obs.size > 0:
        noise_next = rng.normal(0, std, next_obs.shape).astype(np.float32)
        np.clip(next_obs + noise_next, 0.0, 1.0, out=batch['next_observations'])


def apply_board_augment(
    batch: Dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_std: Optional[float] = None,
) -> None:
    """
    Apply random rotation (0/90/180/270°) and horizontal mirror to each sample in the batch.
    Optimized: uses precomputed permutation cache and batched numpy operations.
    Modifies batch in place.
    """
    if batch is None or not isinstance(batch, dict):
        raise ValueError("batch must be a non-empty dict")
    obs = batch.get('observations')
    if obs is None or not isinstance(obs, np.ndarray) or obs.size == 0:
        raise ValueError("batch['observations'] must be a non-empty numpy array")
    gs = batch.get('global_states')
    if gs is None or not isinstance(gs, np.ndarray) or gs.size == 0:
        raise ValueError("batch['global_states'] must be a non-empty numpy array")
    B = obs.shape[0]
    view_size = VIEW_SIZE
    board_size = gs.shape[-1]
    action_size = view_size * view_size

    # Generate all random transforms at once
    rots = rng.integers(0, 4, size=B)
    mirrors = rng.integers(0, 2, size=B)

    for b in range(B):
        rot = int(rots[b])
        mirror = int(mirrors[b])
        if rot == 0 and mirror == 0:
            continue

        # Use precomputed permutations (O(1) lookup instead of 441-iteration loop)
        inv_perm, fwd_perm = _PERM_CACHE[(rot, mirror)]

        # Observations (B, C, H, W) and next_observations
        batch['observations'][b] = _transform_spatial_nd(batch['observations'][b], rot, mirror)
        batch['next_observations'][b] = _transform_spatial_nd(batch['next_observations'][b], rot, mirror)
        # Global states (B, 4, board_size, board_size)
        batch['global_states'][b] = _transform_spatial_nd(batch['global_states'][b], rot, mirror)

        # Actions (B, K): vectorized via fancy indexing
        acts = batch['actions'][b]
        valid = (acts >= 0) & (acts < action_size)
        acts[valid] = fwd_perm[acts[valid]]

        # Target policies (B, K+1, action_size): vectorized via fancy indexing
        batch['target_policies'][b] = batch['target_policies'][b][:, inv_perm]

        # Target center: single int (row*board_size+col)
        flat = int(batch['target_centers'][b])
        row, col = flat // board_size, flat % board_size
        nr, nc = _center_transform(row, col, board_size, rot, mirror)
        batch['target_centers'][b] = nr * board_size + nc

        # Target opponent actions: vectorized
        oa = int(batch['target_opponent_actions'][b])
        if 0 <= oa < action_size:
            batch['target_opponent_actions'][b] = fwd_perm[oa]

        # Target heatmaps (B, 21, 21)
        batch['target_heatmaps'][b] = _transform_spatial_nd(
            batch['target_heatmaps'][b][np.newaxis, ...], rot, mirror
        ).squeeze(0)

    if noise_std is not None and noise_std > 0:
        apply_observation_noise(batch, rng, noise_std)
    return None
