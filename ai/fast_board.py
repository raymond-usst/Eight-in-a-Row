"""Fast board reconstruction — with pure-Python fallback.

Attempts to import the Cython-compiled _fast_board extension.
Falls back to equivalent pure-Python/NumPy implementations if not available.

Usage:
    from ai.fast_board import replay_board_from_snapshot, build_local_obs
"""
import numpy as np

_USE_C = False

try:
    from ai._fast_board import replay_board_from_snapshot as _c_replay
    from ai._fast_board import build_local_obs as _c_build_obs
    _USE_C = True
except ImportError:
    pass


def replay_board_from_snapshot(board, actions, centers_r, centers_c, player_ids,
                                replay_start, replay_end, view_size, board_size):
    """Replay moves onto board (in-place). Uses C extension if available."""
    if _USE_C:
        return _c_replay(board, actions, centers_r, centers_c, player_ids,
                         replay_start, replay_end, view_size, board_size)
    
    # Pure-Python fallback (vectorized where possible)
    half = view_size // 2
    if replay_end - replay_start <= 0:
        return
    
    idx = np.arange(replay_start, replay_end)
    acts = np.asarray(actions)[idx]
    cr = np.asarray(centers_r)[idx]
    cc = np.asarray(centers_c)[idx]
    pids = np.asarray(player_ids)[idx]
    
    lr = acts // view_size
    lc = acts % view_size
    br = cr - half + lr
    bc = cc - half + lc
    
    # Mask valid positions
    valid = (br >= 0) & (br < board_size) & (bc >= 0) & (bc < board_size)
    vbr = br[valid]
    vbc = bc[valid]
    vpids = pids[valid]
    
    # Apply moves (must be sequential for correctness if same cell written twice)
    for i in range(len(vbr)):
        board[vbr[i], vbc[i]] = vpids[i]


def build_local_obs(board, center_r, center_c, current_pid, next_pid, prev_pid,
                    view_size, board_size):
    """Build 4-channel local observation. Uses C extension if available."""
    if _USE_C:
        return _c_build_obs(board, center_r, center_c, current_pid, next_pid, prev_pid,
                            view_size, board_size)
    
    # Pure-Python + NumPy fallback
    half = view_size // 2
    r_start = center_r - half
    c_start = center_c - half
    
    # Pad board to handle boundary
    padded = np.pad(board, pad_width=half, mode='constant', constant_values=-1)
    pr = r_start + half
    pc = c_start + half
    crop = padded[pr:pr + view_size, pc:pc + view_size]
    
    obs = np.zeros((4, view_size, view_size), dtype=np.float32)
    obs[0] = (crop == current_pid).astype(np.float32)
    obs[1] = (crop == next_pid).astype(np.float32)
    obs[2] = (crop == prev_pid).astype(np.float32)
    obs[3] = (crop == 0).astype(np.float32)
    
    return obs
