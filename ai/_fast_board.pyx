"""C-accelerated board reconstruction for replay buffer sampling.

This module provides a Cython-compiled fast path for the inner board replay loop
that reconstructs game positions from snapshots. Falls back to pure Python if
compilation fails.
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy

# Type aliases for clarity
ctypedef np.int8_t INT8
ctypedef np.float32_t FLOAT32


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def replay_board_from_snapshot(
    np.ndarray[INT8, ndim=2] board,
    np.ndarray actions,
    np.ndarray centers_r,
    np.ndarray centers_c,
    np.ndarray player_ids,
    int replay_start,
    int replay_end,
    int view_size,
    int board_size,
):
    """Replay moves from snapshot onto board (in-place).
    
    This is the hot inner loop of sample_batch: for each sampled position,
    we replay ~10-20 moves from the nearest snapshot to reconstruct the board.
    
    Args:
        board: (board_size, board_size) int8 array, modified in-place
        actions: 1D array of action indices
        centers_r: 1D array of center row coordinates
        centers_c: 1D array of center column coordinates
        player_ids: 1D array of player IDs for each step
        replay_start: first step to replay
        replay_end: last step to replay (exclusive)
        view_size: local view size (21)
        board_size: board dimension (100)
    """
    cdef int half = view_size // 2
    cdef int i, act, lr, lc, br, bc, pid
    cdef int cr, cc
    
    for i in range(replay_start, replay_end):
        act = int(actions[i])
        cr = int(centers_r[i])
        cc = int(centers_c[i])
        pid = int(player_ids[i])
        lr = act // view_size
        lc = act % view_size
        br = cr - half + lr
        bc = cc - half + lc
        if 0 <= br < board_size and 0 <= bc < board_size:
            board[br, bc] = pid


@cython.boundscheck(False)
@cython.wraparound(False)
def build_local_obs(
    np.ndarray[INT8, ndim=2] board,
    int center_r,
    int center_c,
    int current_pid,
    int next_pid,
    int prev_pid,
    int view_size,
    int board_size,
):
    """Build 4-channel local observation from board state.
    
    Returns (4, view_size, view_size) float32 array with channels:
      [ME, NEXT, PREV, EMPTY]
    """
    cdef int half = view_size // 2
    cdef np.ndarray[FLOAT32, ndim=3] obs = np.zeros((4, view_size, view_size), dtype=np.float32)
    cdef int r, c, br, bc, val
    
    for r in range(view_size):
        for c in range(view_size):
            br = center_r - half + r
            bc = center_c - half + c
            if 0 <= br < board_size and 0 <= bc < board_size:
                val = board[br, bc]
                if val == current_pid:
                    obs[0, r, c] = 1.0
                elif val == next_pid:
                    obs[1, r, c] = 1.0
                elif val == prev_pid:
                    obs[2, r, c] = 1.0
                elif val == 0:
                    obs[3, r, c] = 1.0
            # else: out-of-bounds → all zeros (empty/wall)
    
    return obs
