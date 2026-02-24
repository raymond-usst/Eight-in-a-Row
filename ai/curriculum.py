"""Curriculum learning: per-stage board size and win length with graduation by games, win rate, loss, and optional Elo.

Maintainability: graduation thresholds and elasticity (curriculum_max_games, base_games) — see ai/MAINTENANCE.md.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ai.log_utils import get_logger

_log = get_logger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Coerce to float; return default on NaN/Inf or invalid."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default

@dataclass
class CurriculumStage:
    stage_id: int
    board_size: int
    win_length: int
    min_steps: int # Fixed absolute step threshold to graduate

class CurriculumManager:
    """
    Per-stage graduation: min_games (scaled by board size), Wilson LCB, loss
    convergence, optional delta-Elo. Elasticity: curriculum_max_games caps
    per-stage min_games; curriculum_base_games sets the 15x15 baseline.
    """
    def __init__(self, config):
        self.config = config

        # Base settings
        self.base_games = getattr(config, 'curriculum_base_games', 10000)
        self.base_size = 15 * 15  # 225

        # Initialize Stages with Fixed Step Counts: 500 -> 1000 -> 1500 -> infinity
        self.stages = [
            CurriculumStage(stage_id=1, board_size=15, win_length=5, min_steps=500),
            CurriculumStage(stage_id=2, board_size=30, win_length=6, min_steps=1000),
            CurriculumStage(stage_id=3, board_size=50, win_length=7, min_steps=1500),
            CurriculumStage(stage_id=4, board_size=100, win_length=8, min_steps=99999999)
        ]
        
        self.current_stage_idx = 0
        self.games_in_stage = 0
        
        # Stats Buffers (kept for dashboard/logging info)
        self.win_rate_buffer = []   
        self.loss_buffer = []       
        self._loss_ema = None       
        self.loss_ema_buffer = []   
        self.use_ema_loss = getattr(config, 'curriculum_use_ema_loss', True)
        self.ema_alpha = getattr(config, 'curriculum_ema_alpha', 0.1)
        
        self.stage_start_elo = None

    def set_stage(self, stage_idx: int):
        if 0 <= stage_idx < len(self.stages):
            self.current_stage_idx = stage_idx
            self.games_in_stage = 0
            self.win_rate_buffer = []
            self.loss_buffer = []
            self.loss_ema_buffer = []
            self._loss_ema = None
            self.stage_start_elo = None
            _log.info("Manually set to Stage %d: %s", stage_idx + 1, self.get_current_stage())

    def get_current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]
        
    def record_game_result(self, score: float):
        """
        Record one game result for graduation stats. Call only when a game finishes.
        score: 1.0 win, 0.5 draw, 0.0 loss (for the current/active agent).
        """
        s = _safe_float(score, default=-1.0)
        if s < 0 or s > 1:
            return  # skip invalid; optional: log once
        self.games_in_stage += 1
        self.win_rate_buffer.append(s)
        if len(self.win_rate_buffer) > 1000:
            self.win_rate_buffer.pop(0)

        # Probation removed
        pass

    def record_loss(self, loss: float):
        """
        Record training loss for convergence check. Does not touch games_in_stage or win_rate_buffer.
        Invalid/NaN/Inf loss is skipped.
        """
        v = _safe_float(loss, default=float('nan'))
        if math.isnan(v):
            return
        self.loss_buffer.append(v)
        if len(self.loss_buffer) > 2000:
            self.loss_buffer.pop(0)
        if self.use_ema_loss:
            self._loss_ema = v if self._loss_ema is None else (
                (1.0 - self.ema_alpha) * self._loss_ema + self.ema_alpha * v
            )
            self.loss_ema_buffer.append(self._loss_ema)
            if len(self.loss_ema_buffer) > 2000:
                self.loss_ema_buffer.pop(0)

    def update_stats(self, win_rate_vs_frozen: float, loss: float):
        """
        Legacy: update both game result and loss. Prefer record_game_result + record_loss separately.
        """
        self.record_game_result(win_rate_vs_frozen)
        self.record_loss(loss)
            
    def check_graduation(self, step: int, league=None) -> bool:
        """
        Check if we should advance to the next stage using absolute training steps.
        """
        stage = self.get_current_stage()
        
        # 1. Max Stage check
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        # 2. Check fixed step threshold
        if step >= stage.min_steps:
            _log.info("Graduation triggered: Stage %s complete at step %d (>= %d).", stage.stage_id, step, stage.min_steps)
            return True
            
        return False
        
    def advance(self, league=None):
        """Advance to next stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.games_in_stage = 0
            self.win_rate_buffer = []
            self.loss_buffer = []
            self.loss_ema_buffer = []
            self._loss_ema = None
            if league is not None:
                self.stage_start_elo = getattr(league, 'current_elo', None)
            
            _log.info("Advanced to Stage %d.", self.current_stage_idx + 1)
            return self.get_current_stage()
        return None

    def state_dict(self) -> Dict[str, Any]:
        """Return the current state of the curriculum."""
        return {
            'current_stage_idx': self.current_stage_idx,
            'games_in_stage': self.games_in_stage,
            'win_rate_buffer': self.win_rate_buffer,
            'loss_buffer': self.loss_buffer,
            'stage_start_elo': self.stage_start_elo,
            'loss_ema_buffer': getattr(self, 'loss_ema_buffer', []),
            '_loss_ema': getattr(self, '_loss_ema', None),
        }

    def _ensure_number_list(self, val: Any, name: str, max_len: int = 5000) -> List[float]:
        """Return a list of finite numbers; else empty list (with optional warning)."""
        if not isinstance(val, list):
            return []
        out = []
        for x in val:
            try:
                f = float(x)
                if math.isfinite(f):
                    out.append(f)
            except (TypeError, ValueError):
                continue
        if len(out) != len(val) and val:
            _log.warning("%s had invalid entries, using %d valid.", name, len(out))
        return out[:max_len]

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore curriculum state. Backward compatible with older checkpoints missing new keys."""
        self.current_stage_idx = int(state_dict.get('current_stage_idx', 0))
        self.games_in_stage = int(state_dict.get('games_in_stage', 0))
        self.win_rate_buffer = self._ensure_number_list(
            state_dict.get('win_rate_buffer'), 'win_rate_buffer', max_len=1000
        )
        self.loss_buffer = self._ensure_number_list(
            state_dict.get('loss_buffer'), 'loss_buffer', max_len=2000
        )
        self.stage_start_elo = state_dict.get('stage_start_elo', None)
        self.loss_ema_buffer = self._ensure_number_list(
            state_dict.get('loss_ema_buffer'), 'loss_ema_buffer', max_len=2000
        )
        self._loss_ema = state_dict.get('_loss_ema', None)
        if self._loss_ema is not None:
            try:
                self._loss_ema = float(self._loss_ema)
                if not math.isfinite(self._loss_ema):
                    self._loss_ema = None
            except (TypeError, ValueError):
                self._loss_ema = None
        # Probation state (removed, kept for compatibility if needed)
        self.current_stage_idx = max(0, min(self.current_stage_idx, len(self.stages) - 1))
        _log.info("State loaded: Stage %d, Games %d",
                  self.current_stage_idx + 1, self.games_in_stage)
