"""Self-play game generation for MuZero training.

Maintainability: see ai/MAINTENANCE.md."""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Union, Any
from .game_env import EightInARowEnv
from .mcts import gumbel_muzero_search, select_action
from .replay_buffer import GameHistory
from .log_utils import get_logger

_log = get_logger(__name__)


@dataclass
class SessionContext:
    """Tracks cumulative scores across a multi-game session."""
    scores: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    game_idx: int = 0
    session_length: int = 1

    def make_context_vector(self, current_pid: int) -> np.ndarray:
        """Create a 4-dim normalized context vector for the current player.

        Returns [my_score_norm, opp1_score_norm, opp2_score_norm, games_remaining_norm].
        All values in [0, 1].
        """
        max_possible = max(1, self.session_length * 5)  # max score = winning every game
        pids = sorted(self.scores.keys())
        others = [p for p in pids if p != current_pid]

        my_score = self.scores.get(current_pid, 0) / max_possible
        opp1_score = self.scores.get(others[0], 0) / max_possible if len(others) > 0 else 0.0
        opp2_score = self.scores.get(others[1], 0) / max_possible if len(others) > 1 else 0.0
        games_remaining = (self.session_length - self.game_idx - 1) / max(1, self.session_length)

        return np.array([my_score, opp1_score, opp2_score, games_remaining], dtype=np.float32)


def _predict_view_center(env, network, config, device, rotated):
    """Predict view center using Focus Network with smart-center fallback.

    Returns (center_r, center_c, legal_moves_list, legal_mask, current_net).
    """
    gs_tensor = torch.from_numpy(rotated[np.newaxis, ...]).to(device)

    if isinstance(network, dict):
        current_net = network[env.current_player_id]
    else:
        current_net = network

    try:
        cr, cc = current_net.predict_center(gs_tensor)
    except (AttributeError, ValueError) as e:
        if isinstance(e, ValueError) and 'NaN' in str(e):
            _log.error("CRITICAL: NaN detected in predict_center! Saving crash dump...")
            import os
            debug_path = f"debug_crash_actor_{os.getpid()}.pt"
            torch.save({
                'global_state': gs_tensor.cpu(),
                'network_state': network.state_dict(),
                'focus_net_buffers': dict(network.focus_net.named_buffers())
            }, debug_path)
            _log.info("Saved crash dump to %s", debug_path)
            raise RuntimeError("ModelCorrupted")
        cr, cc = env.get_smart_center(config.local_view_size)

    # Validate center has legal moves
    legal_check, legal_mask = env.get_legal_moves_and_mask(cr, cc, config.local_view_size)
    if len(legal_check) == 0:
        cr, cc = env.get_smart_center(config.local_view_size)
        legal_check, legal_mask = env.get_legal_moves_and_mask(cr, cc, config.local_view_size)

    return cr, cc, legal_check, legal_mask, current_net


def _get_observation_and_mask(env, config, cr, cc, legal_mask, rotated):
    """Get observation and fix legal mask if center was clamped.

    Returns (obs, center, legal_mask).
    """
    obs, center = env.get_observation(config.local_view_size, center=(cr, cc),
                                       rotated_planes=rotated)
    if center != (cr, cc):
        _, legal_mask = env.get_legal_moves_and_mask(center[0], center[1], config.local_view_size)
    return obs, center, legal_mask


def _validate_and_step(env, action, center, config):
    """Validate action on the board and execute. Falls back to random legal move if needed.

    Returns (board_r, board_c, action_was_legal, reward, done).
    """
    board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)
    action_was_legal = True

    if board_r < 0 or board_r >= env.BOARD_SIZE or board_c < 0 or board_c >= env.BOARD_SIZE:
        legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
        if len(legal) == 0:
            return board_r, board_c, False, 0.0, True  # signal break
        board_r, board_c = legal[np.random.randint(len(legal))]
        action_was_legal = False
    elif env.board[board_r, board_c] != 0:
        legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
        if len(legal) == 0:
            return board_r, board_c, False, 0.0, True  # signal break
        board_r, board_c = legal[np.random.randint(len(legal))]
        action_was_legal = False

    reward, done = env.step(board_r, board_c)
    return board_r, board_c, action_was_legal, reward, done


def play_game(network: Union[torch.nn.Module, Dict[int, torch.nn.Module]], config,
              temperature: float = 1.0,
              broadcast_fn: Optional[Callable] = None,
              game_index: int = 0,
              iteration: int = 0,
              session_context: Optional[SessionContext] = None,
              training_step: int = 0,
              board_size_override: Optional[int] = None,
              win_length_override: Optional[int] = None,
              pbt_elo: Optional[float] = None) -> GameHistory:
    """
    Play one complete game using MCTS.
    network: either a single MuZeroNetwork (shared) or a dict {pid: net} (KOTH mode).
    broadcast_fn(event_type, data): optional callback for live visualization.
    Returns a GameHistory with the full trajectory.

    Config (or config-like object) must have: board_size, win_length, num_players,
    local_view_size, temperature_drop_step, num_simulations, max_game_steps;
    optional: num_simulations_early/mid/late, policy_target_temp_start/end/steps.
    See ai/REUSABILITY.md.
    """
    board_size = board_size_override if board_size_override is not None else config.board_size
    win_length = win_length_override if win_length_override is not None else config.win_length

    # Cap game length by board size to avoid GPU OOM and recompile thrashing (100x100 -> 250, etc.)
    max_game_steps_cfg = getattr(config, 'max_game_steps', 5000)
    if board_size >= 50:
        max_steps_env = min(max_game_steps_cfg, 250)
    elif board_size >= 30:
        max_steps_env = min(max_game_steps_cfg, 180)
    else:
        max_steps_env = min(max_game_steps_cfg, 150)

    env = EightInARowEnv(board_size=board_size, win_length=win_length, max_steps=max_steps_env)
    env.reset()
    history = GameHistory()
    history.board_size = board_size
    history.win_length = win_length

    if isinstance(network, dict):
        device = next(network[1].parameters()).device
        for net in network.values():
            net.eval()
    else:
        network.eval()
        device = next(network.parameters()).device  # cache once, reuse every move

    # Level 4: Asymmetric exploration — one random player explores more aggressively
    explorer_pid = np.random.randint(1, config.num_players + 1)  # random player 1-3
    explorer_noise_scale = 2.0   # 2x Gumbel noise for explorer
    explorer_epsilon = 0.1       # 10% random actions for explorer

    # Broadcast game start
    if broadcast_fn:
        broadcast_fn('selfplay_start', {
            'game_index': game_index,
            'iteration': iteration,
        })

    # Efficiency: cache config lookups once per game to avoid getattr in the step loop.
    temperature_drop_step = getattr(config, 'temperature_drop_step', 0)
    progress = min(1.0, training_step / max(1, getattr(config, 'progression_steps', 100000)))
    sims_start = getattr(config, 'num_simulations_start', 25)
    sims_end = getattr(config, 'num_simulations_end', 200)
    base_sims = int(sims_start + (sims_end - sims_start) * progress)
    
    sims_early = getattr(config, 'num_simulations_early', max(8, base_sims // 3))
    sims_mid = getattr(config, 'num_simulations_mid', base_sims)
    sims_late = getattr(config, 'num_simulations_late', max(16, base_sims // 2))
    policy_target_temp_start = getattr(config, 'policy_target_temp_start', 1.0)
    policy_target_temp_end = getattr(config, 'policy_target_temp_end', 1.0)
    policy_target_temp_steps = getattr(config, 'policy_target_temp_steps', 100000)
    max_game_steps = getattr(config, 'max_game_steps', 5000)

    step = 0
    step_retries = 0
    max_step_retries = 2
    previous_root = None  # MCTS tree reuse: track subtree from previous search
    
    # -------------------------------------------------------------
    # Exponential Temperature Decay & Elo-Modulation
    # -------------------------------------------------------------
    base_temp = temperature
    
    # Global training-step based exploration annealing
    explore_start = getattr(config, 'explore_temp_start', temperature)
    explore_end = getattr(config, 'explore_temp_end', 0.2)
    explore_steps = getattr(config, 'explore_temp_steps', 200000)
    
    if explore_steps > 0 and explore_start != explore_end:
        progress = min(1.0, training_step / explore_steps)
        base_temp = explore_start + (explore_end - explore_start) * progress
    if pbt_elo is not None:
        # PBT dynamic feedback: if losing (Elo drop), increase exploration. If mastering, exploit.
        if pbt_elo < 1200:
            base_temp = min(1.25, base_temp + (1200 - pbt_elo) * 0.001)
        elif pbt_elo > 1600:
            base_temp = max(0.85, base_temp - (pbt_elo - 1600) * 0.0005)
            
    # Calculate exponential decay factor to reach exactly 0.1 at temperature_drop_step
    decay_factor = 1.0
    if temperature_drop_step > 0:
        decay_factor = (0.1 / base_temp) ** (1.0 / temperature_drop_step)
    
    while not env.done:
        # Determine temperature via exponential decay
        if step < temperature_drop_step:
            temp = base_temp * (decay_factor ** step)
        else:
            temp = 0.1

        # 1. Predict view center + get observation
        rotated = env._get_rotated_planes_cached()
        cr, cc, _, legal_mask, current_net = _predict_view_center(
            env, network, config, device, rotated
        )
        obs, center, legal_mask = _get_observation_and_mask(
            env, config, cr, cc, legal_mask, rotated
        )

        # Session context for current player
        ctx_vec = None
        if session_context is not None:
            ctx_vec = session_context.make_context_vector(env.current_player_id)

        # Dynamic simulation budget
        dynamic_sims = sims_early if step < 10 else (sims_mid if step < 40 else sims_late)

        # Asymmetric noise
        current_pid = env.current_player_id
        ns = explorer_noise_scale if current_pid == explorer_pid else 1.0

        # Single-step try/except: MCTS + env.step; on failure log, retry once, then return partial history
        try:
            # 2. Run MCTS
            action_probs, root_value, root = gumbel_muzero_search(
                current_net, obs, legal_mask, config,
                add_noise=True,
                noise_scale=ns,
                session_context_vec=ctx_vec,
                device=device,
                num_simulations_override=dynamic_sims,
                reuse_tree=previous_root
            )

            # Policy target temperature annealing
            t_start = policy_target_temp_start
            t_end = policy_target_temp_end
            t_steps = policy_target_temp_steps
            if t_start != t_end and t_steps > 0:
                progress = min(1.0, training_step / t_steps)
                policy_temp = t_start + (t_end - t_start) * progress
                target_probs = action_probs ** (1.0 / policy_temp)
                t_sum = target_probs.sum()
                target_probs = target_probs / t_sum if t_sum > 0 else action_probs
            else:
                target_probs = action_probs

            # Select action
            action = select_action(action_probs, temperature=temp)

            # Explorer epsilon-greedy
            player_id = env.current_player_id
            if player_id == explorer_pid and np.random.rand() < explorer_epsilon:
                legal_indices = np.where(legal_mask > 0)[0]
                if len(legal_indices) > 0:
                    action = int(np.random.choice(legal_indices))

            # Threats + store in history
            threats = env.get_threat_levels()
            history.store(action, 0.0, target_probs, root_value, threats, player_id, center)

            # 3. Validate and execute action
            board_r, board_c, action_was_legal, reward, should_break = _validate_and_step(
                env, action, center, config
            )
            if should_break and not env.done:
                break

            if not action_was_legal and len(history.actions) > 0:
                fallback_action = env.board_to_action(board_r, board_c, center[0], center[1], config.local_view_size)
                history.actions[-1] = fallback_action

            # Broadcast the move
            if broadcast_fn:
                broadcast_fn('selfplay_move', {
                    'row': int(board_r),
                    'col': int(board_c),
                    'player': int(player_id),
                    'step': step,
                })

            # (env.step already called inside _validate_and_step)

            # Save board snapshot periodically for fast replay buffer reconstruction
            if GameHistory.SNAPSHOT_INTERVAL > 0 and step % GameHistory.SNAPSHOT_INTERVAL == 0:
                history.board_snapshots[step] = env.board.copy()

            # Update reward for the move that just happened
            history.rewards[-1] = reward
            step += 1
            step_retries = 0

            # MCTS tree reuse: extract child subtree for next search
            # Discard if action was overridden (fallback) since the tree is invalid
            if action_was_legal and root is not None:
                previous_root = root.children.get(action, None)
            else:
                previous_root = None
        except Exception as e:
            step_retries += 1
            _log.warning(
                "Step failed (retry %d/%d): %s: %s",
                step_retries, max_step_retries, type(e).__name__, e
            )
            if step_retries >= max_step_retries:
                _log.warning("Consecutive step failures; returning partial history.")
                break

        # Safety: prevent absurdly long games (configurable via config.max_game_steps; see SCALABILITY.md)
        if step > max_game_steps:
            break

    history.done = env.done
    history.winner = env.winner
    history.rankings = list(env.rankings)
    history.placement_rewards = dict(env.placement_rewards)
    history.final_board = env.board.copy()

    # Store session-level context for Phase 4 network awareness
    if session_context is not None:
        history.session_scores = dict(session_context.scores)
        history.session_game_idx = session_context.game_idx
        history.session_length = session_context.session_length

    # ── Terminal reward shaping using placement-based rewards ──
    # 1st: +1.0, 2nd: -0.2, 3rd: -1.0  (from 5:2:0 point mapping)
    # Assign decaying rewards to each player's last moves before game end.
    if env.winner is not None and env.placement_rewards:
        # Track which players have already received their terminal shaping
        shaped = set()
        for i in range(len(history) - 1, -1, -1):
            pid = history.player_ids[i]
            if pid not in shaped:
                base_reward = env.placement_rewards.get(pid, -1.0)
                distance = len(history) - 1 - i  # moves before end
                decay = 0.8 ** distance  # strongest at last move, decaying backwards
                history.rewards[i] = base_reward * decay
                shaped.add(pid)
            if len(shaped) >= env.NUM_PLAYERS:
                break  # All players shaped

    # Broadcast game end (with ranking & session info for dashboard)
    if broadcast_fn:
        evt = {
            'game_index': game_index,
            'winner': int(history.winner) if history.winner else None,
            'length': step,
            'rankings': [(int(pid), int(pl)) for pid, pl in env.rankings] if env.rankings else [],
        }
        if session_context is not None:
            evt['session'] = {
                'scores': {int(k): int(v) for k, v in session_context.scores.items()},
                'game_idx': session_context.game_idx,
                'session_length': session_context.session_length,
            }
        broadcast_fn('selfplay_end', evt)
        
    return history


def play_session(network: torch.nn.Module, config,
                 session_length: int,
                 temperature: float = 1.0,
                 broadcast_fn: Optional[Callable] = None,
                 game_index_base: int = 0,
                 iteration: int = 0,
                 training_step: int = 0,
                 board_size_override: Optional[int] = None,
                 win_length_override: Optional[int] = None,
                 pbt_elo: Optional[float] = None) -> List[GameHistory]:
    """Play a Best-of-5 round: first player to 3 wins is the round champion.

    Remaining players are ranked by cumulative score (冠军+3, 亚军+0, 季军-2).
    Scores reset each round. The round terminates early when a player reaches
    3 wins.

    Each GameHistory carries round metadata:
        round_winner: pid of the round champion (or None if not yet decided)
        round_rankings: [(pid, placement)] for the round
        round_scores: {pid: cumulative_score} at end of round

    Args:
        session_length: Number of games in this round (typically 5).
        game_index_base: Base index for game numbering (for logging).

    Returns:
        List of GameHistory objects, one per game played in the round.
    """
    scores: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    wins: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    histories: List[GameHistory] = []
    round_winner = None
    wins_needed = 3  # Best-of-5

    for game_idx in range(session_length):
        context = SessionContext(
            scores=dict(scores),  # snapshot current scores
            game_idx=game_idx,
            session_length=session_length,
        )

        game = play_game(
            network, config,
            temperature=temperature,
            broadcast_fn=broadcast_fn,
            game_index=game_index_base + game_idx,
            iteration=iteration,
            session_context=context,
            training_step=training_step,
            board_size_override=board_size_override,
            win_length_override=win_length_override,
            pbt_elo=pbt_elo,
        )

        # Update cumulative scores based on game rankings
        for pid, placement in game.rankings:
            scores[pid] += config.placement_points[placement]

        # Track per-game wins (1st place = win)
        if game.rankings:
            game_champion = game.rankings[0][0]
            wins[game_champion] += 1

        histories.append(game)

        # Check for early termination: any player reached 3 wins
        for pid, w in wins.items():
            if w >= wins_needed:
                round_winner = pid
                break
        if round_winner is not None:
            break

    # Determine round rankings
    round_rankings = []
    if round_winner is not None:
        round_rankings.append((round_winner, 0))  # Champion
        # Rank remaining players by cumulative score (descending)
        others = [(pid, scores[pid]) for pid in [1, 2, 3] if pid != round_winner]
        others.sort(key=lambda x: x[1], reverse=True)
        for i, (pid, _) in enumerate(others):
            round_rankings.append((pid, i + 1))  # Runner-up, Third
    else:
        # No player reached 3 wins (session shorter than 5 games in early training):
        # rank all by cumulative score, top scorer is round winner
        all_players = [(pid, scores[pid]) for pid in [1, 2, 3]]
        all_players.sort(key=lambda x: x[1], reverse=True)
        round_winner = all_players[0][0]
        for i, (pid, _) in enumerate(all_players):
            round_rankings.append((pid, i))

    # Ensure exactly 3 entries for dashboard (Champion, Runner-up, Third); placement 0/1/2
    assert len(round_rankings) == 3, "round_rankings must have 3 entries for dashboard"

    # Attach round metadata to all games in this round
    for h in histories:
        h.round_winner = round_winner
        h.round_rankings = round_rankings
        h.round_scores = dict(scores)

    # Broadcast round end
    if broadcast_fn:
        broadcast_fn('round_end', {
            'round_winner': int(round_winner) if round_winner else None,
            'round_rankings': [(int(pid), int(pl)) for pid, pl in round_rankings],
            'round_scores': {int(k): int(v) for k, v in scores.items()},
            'games_played': len(histories),
            'wins': {int(k): int(v) for k, v in wins.items()},
        })

    return histories


def get_adaptive_session_length(training_step: int, config) -> int:
    """Fixed Best-of-5 session length."""
    return 5


def run_selfplay(network: torch.nn.Module, config, num_games: int, 
                 broadcast_fn: Optional[Callable] = None,
                 iteration: int = 0) -> List[GameHistory]:
    """Run multiple self-play games."""
    # For simplicity, run sequentially. Parallelization is harder with single GPU network.
    games = []
    for i in range(num_games):
        games.append(play_game(network, config, 
                               broadcast_fn=broadcast_fn, 
                               game_index=i+1,
                               iteration=iteration))
    return games
