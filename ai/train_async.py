"""Async MuZero training: CPU Self-Play + GPU Training."""

import os
import re
import sys
import time
import json
import queue
import torch
import torch.multiprocessing as mp
import numpy as np
import traceback
import argparse
import asyncio
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.replay_buffer import ReplayBuffer
from ai.self_play import play_game, play_session, get_adaptive_session_length
from ai.engram import MemoryBank
from ai.train import train_step, save_checkpoint, WS_CLIENTS, WS_LOOP, start_ws_server, broadcast, save_metrics_log, load_metrics_log, METRICS_LOG
from ai.pbt import Population
from ai.curriculum import CurriculumManager
from ai.league import LeagueManager
from ai.ipc_inference import InferenceClient, InferenceServer
from ai.board_render import board_to_image_path
from ai.data_augment import apply_board_augment
from ai.log_utils import get_logger

_log = get_logger(__name__)
_log_learner = get_logger("ai.train_async.learner")

# Windows compatibility for multiprocessing
# We need to set start method to 'spawn' in main block, but mp.set_start_method is global.
# We will do it in main.

def _strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix that torch.compile() adds to state_dict keys.

    When a model is wrapped with torch.compile(), its state_dict keys get
    prefixed with '_orig_mod.'. This helper strips that prefix so the state
    can be loaded into a non-compiled model (e.g. shared_model, file checkpoint).
    """
    prefix = '_orig_mod.'
    needs_strip = any(k.startswith(prefix) for k in state_dict)
    if not needs_strip:
        return state_dict
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def _ensure_frozen_state_compatible(state_dict, ref_keys, current_sd, pid_label):
    """Ensure a state_dict has the same keys as the current model; else use current_sd and log."""
    if set(state_dict.keys()) == ref_keys:
        return {k: v.clone() for k, v in state_dict.items()}
    print(f"[Learner] KOTH: {pid_label} incompatible keys, using current model weights.")
    return {k: v.clone() for k, v in current_sd.items()}


def _sanitize_frozen_models_for_save(frozen_models, model):
    """Replace any frozen_models slot whose keys don't match current model (before saving to disk)."""
    current_sd = _strip_compiled_prefix(model.state_dict())
    ref_keys = set(current_sd.keys())
    for pid in [1, 2, 3]:
        if pid in frozen_models and set(frozen_models[pid].keys()) != ref_keys:
            frozen_models[pid] = {k: v.clone() for k, v in current_sd.items()}


def atomic_torch_save(obj, path):
    """Atomic save: write to .tmp then os.replace() to avoid corruption on interrupt."""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    for i in range(10):
        try:
            if os.path.exists(path):
                os.replace(tmp_path, path)
            else:
                os.rename(tmp_path, path)
            break
        except PermissionError:
            if i < 9:
                time.sleep(0.1)
            else:
                raise

class SharedStats:
    """Multiprocessing-safe counters for games, steps, placements, win counts; shared between actors and learner."""
    def __init__(self):
        self.total_games = mp.Value('i', 0)
        self.total_steps = mp.Value('i', 0)
        self.total_game_len = mp.Value('i', 0)
        self.win_counts = mp.Array('i', 4) # red, green, blue, draw
        # Placement tracking: [p1_1st, p1_2nd, p1_3rd, p2_1st, p2_2nd, p2_3rd, p3_1st, p3_2nd, p3_3rd]
        self.placement_counts = mp.Array('i', 9)
        self.ranked_games = mp.Value('i', 0)
        # Round-level tracking (Best-of-5)
        self.total_rounds = mp.Value('i', 0)
        self.round_win_counts = mp.Array('i', 4)  # red, green, blue, draw
        self.round_placement_counts = mp.Array('i', 9)  # same layout as placement_counts
        self.lock = mp.Lock()

    def update_game(self, winner, length, rankings=None):
        with self.lock:
            self.total_games.value += 1
            self.total_game_len.value += length
            if winner is None:
                self.win_counts[3] += 1
            elif 1 <= winner <= 3:
                self.win_counts[winner-1] += 1
            # Update placement counts from rankings [(pid, placement), ...]
            if rankings:
                self.ranked_games.value += 1
                for pid, placement in rankings:
                    if 1 <= pid <= 3 and 0 <= placement <= 2:
                        idx = (pid - 1) * 3 + placement
                        self.placement_counts[idx] += 1

    def update_round(self, round_winner, round_rankings):
        """Update round-level stats after a Best-of-5 round completes."""
        with self.lock:
            self.total_rounds.value += 1
            if round_winner is None:
                self.round_win_counts[3] += 1
            elif 1 <= round_winner <= 3:
                self.round_win_counts[round_winner - 1] += 1
            if round_rankings:
                for pid, placement in round_rankings:
                    if 1 <= pid <= 3 and 0 <= placement <= 2:
                        idx = (pid - 1) * 3 + placement
                        self.round_placement_counts[idx] += 1

    def get_info(self):
        with self.lock:
            avg_len = 0
            if self.total_games.value > 0:
                avg_len = self.total_game_len.value / self.total_games.value
            return {
                'games': self.total_games.value,
                'steps': self.total_steps.value,
                'avg_len': avg_len,
                'total_len': self.total_game_len.value,
                'wins': {
                    '1': self.win_counts[0],
                    '2': self.win_counts[1],
                    '3': self.win_counts[2],
                    'draw': self.win_counts[3]
                },
                'ranked_games': self.ranked_games.value,
                'placements': {
                    '1': [self.placement_counts[0], self.placement_counts[1], self.placement_counts[2]],
                    '2': [self.placement_counts[3], self.placement_counts[4], self.placement_counts[5]],
                    '3': [self.placement_counts[6], self.placement_counts[7], self.placement_counts[8]],
                },
                'total_rounds': self.total_rounds.value,
                'round_wins': {
                    '1': self.round_win_counts[0],
                    '2': self.round_win_counts[1],
                    '3': self.round_win_counts[2],
                    'draw': self.round_win_counts[3],
                },
                'round_placements': {
                    '1': [self.round_placement_counts[0], self.round_placement_counts[1], self.round_placement_counts[2]],
                    '2': [self.round_placement_counts[3], self.round_placement_counts[4], self.round_placement_counts[5]],
                    '3': [self.round_placement_counts[6], self.round_placement_counts[7], self.round_placement_counts[8]],
                },
            }

# CurriculumScheduler removed in favor of CurriculumManager

def reanalyze_worker(rank, config, reanalyze_in_queue, reanalyze_out_queue, stop_event, shared_model, weights_version, weights_lock):
    """Background process that evaluates old games with the latest network."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Needs GPU
    import torch
    import numpy as np
    from ai.muzero_network import MuZeroNetwork
    from ai.game_env import EightInARowEnv
    from ai.mcts import gumbel_muzero_search
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = MuZeroNetwork(config).to(device)
    network.eval()
    local_version = -1
    h_half = config.local_view_size // 2

    while not stop_event.is_set():
        # Sync weights
        if weights_version.value > local_version:
            with weights_lock:
                network.load_state_dict(shared_model.state_dict())
                local_version = weights_version.value
                
        try:
            insert_idx, game = reanalyze_in_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        try:
            env = EightInARowEnv(board_size=getattr(game, 'board_size', config.board_size), 
                                 win_length=getattr(game, 'win_length', config.win_length))
            env.reset()
            
            new_policies, new_values = [], []
            
            for i in range(len(game)):
                ctr = game.centers[i]
                from ai.self_play import _get_observation_and_mask
                rotated = env._get_rotated_planes_cached()
                _, legal_mask = env.get_legal_moves_and_mask(ctr[0], ctr[1], config.local_view_size)
                obs, _, _ = _get_observation_and_mask(env, config, ctr[0], ctr[1], legal_mask, rotated)
                
                ctx = None
                if game.session_scores is not None:
                    pid = game.player_ids[i]
                    max_possible = max(1, game.session_length * 5)
                    pids_sorted = sorted(game.session_scores.keys())
                    others = [p for p in pids_sorted if p != pid]
                    my_score = game.session_scores.get(pid, 0) / max_possible
                    opp1_score = game.session_scores.get(others[0], 0) / max_possible if len(others) > 0 else 0.0
                    opp2_score = game.session_scores.get(others[1], 0) / max_possible if len(others) > 1 else 0.0
                    games_remaining = (game.session_length - game.session_game_idx - 1) / max(1, game.session_length)
                    ctx = np.array([my_score, opp1_score, opp2_score, games_remaining], dtype=np.float32)

                action_probs, root_value, _ = gumbel_muzero_search(
                    network, obs, legal_mask, config, add_noise=False, 
                    session_context_vec=ctx, device=device
                )
                
                new_policies.append(action_probs)
                new_values.append(root_value)
                
                act = game.actions[i]
                lr, lc = act // config.local_view_size, act % config.local_view_size
                env.step(ctr[0] - h_half + lr, ctr[1] - h_half + lc)
                
            game.policy_targets = new_policies
            game.root_values = new_values
            
            try:
                reanalyze_out_queue.put((insert_idx, game), timeout=1.0)
            except queue.Full:
                pass
                
        except Exception as e:
            _log.error("[Reanalyze Worker] Error updating game: %s", e)

def buffer_loop(config, buffer_queue, batch_queue, cmd_queue, res_queue, stop_event, shared_buffer_games,
                shared_stats=None, reanalyze_in_queue=None, reanalyze_out_queue=None):
    """Background process for maintaining the ReplayBuffer and sampling batches."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No GPU needed
    try:
        from ai.replay_buffer import ReplayBuffer
        from ai.data_augment import apply_board_augment
        import numpy as np
        
        buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
            max_memory_gb=config.max_memory_gb,
            priority_alpha=config.priority_alpha,
            min_games=config.min_buffer_games
        )

        def batch_worker():
            numpy_rng = np.random.default_rng()
            while not stop_event.is_set():
                try:
                    if buffer.num_games() >= config.min_buffer_size:
                        t_step = shared_stats.total_steps.value if shared_stats else 0
                        progress = min(1.0, t_step / max(1, config.progression_steps))
                        current_batch_size = int(config.batch_size_start + (config.batch_size_end - config.batch_size_start) * progress)

                        batch = buffer.sample_batch(
                            batch_size=current_batch_size,
                            num_unroll_steps=config.num_unroll_steps,
                            td_steps=config.td_steps,
                            discount=config.discount,
                            action_size=config.policy_size,
                            view_size=config.local_view_size
                        )
                        if getattr(config, 'augment_board', False):
                            apply_board_augment(batch, numpy_rng, noise_std=getattr(config, 'augment_noise_std', 0.0) or None)
                        batch_queue.put(batch, timeout=1.0)
                    else:
                        time.sleep(0.5)
                except queue.Full:
                    time.sleep(0.1)
                except Exception as e:
                    _log.error("[Buffer Thread] Error sampling batch: %s", e)
                    time.sleep(0.5)

        bt = threading.Thread(target=batch_worker, daemon=True)
        bt.start()

        last_games_update = 0
        last_reanalyze_push = 0
        while not stop_event.is_set():
            for _ in range(50):
                try:
                    game, step = buffer_queue.get_nowait()
                    buffer.save_game(game, training_step=step)
                except queue.Empty:
                    break
            
            # Process reanalyze outputs
            if reanalyze_out_queue:
                for _ in range(10):
                    try:
                        insert_idx, upgraded_game = reanalyze_out_queue.get_nowait()
                        buffer.update_game_by_insert_idx(insert_idx, upgraded_game)
                    except queue.Empty:
                        break
            
            # Feed reanalyze worker occasionally (1 game/sec approx)
            now = time.time()
            if reanalyze_in_queue and (now - last_reanalyze_push > 1.0):
                if buffer.num_games() >= config.min_buffer_size and not reanalyze_in_queue.full():
                    sampled = buffer.sample_game_for_reanalyze()
                    if sampled:
                        try:
                            reanalyze_in_queue.put_nowait(sampled)
                        except queue.Full:
                            pass
                last_reanalyze_push = now
            
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                    cmd_type = cmd.get('type')
                    if cmd_type == 'update_priorities':
                        buffer.update_priorities(cmd['indices'], cmd['errors'])
                    elif cmd_type == 'clear':
                        buffer.clear()
                        res_queue.put({'type': 'clear_ok'})
                    elif cmd_type == 'load':
                        buffer.load(cmd['path'])
                        res_queue.put({'type': 'load_ok', 'num_games': buffer.num_games()})
                    elif cmd_type == 'save':
                        buffer.save(cmd['path'])
                        res_queue.put({'type': 'save_ok', 'report': buffer.memory_report()})
                    elif cmd_type == 'report':
                        res_queue.put({'type': 'report_ok', 'report': buffer.memory_report()})
                    elif cmd_type == 'num_games':
                        res_queue.put({'type': 'num_games_ok', 'num_games': buffer.num_games()})
                except queue.Empty:
                    break
            
            # Update shared count more frequently (0.2s) so Data Gen progress displays correctly
            if now - last_games_update > 0.2:
                shared_buffer_games.value = buffer.num_games()
                last_games_update = now

            time.sleep(0.01)
    except Exception as e:
        _log.error("[Buffer Process] Crash: %s", e)
        traceback.print_exc()


def actor_loop(rank, config, game_queue, weights_path, memory_path, stop_event, shared_stats,
               live_queue=None, shared_model=None, weights_version=None, weights_lock=None, shared_curriculum_stage=None):
    """Actor process: runs on CPU, generates games."""
    # Loop: (1) load weights from shared memory or file, (2) optionally load curriculum/league/KOTH state, (3) play game(s), (4) push to game_queue.
    try:
        # GPU visibility disable for actor
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Initialize network on CPU
        model = MuZeroNetwork(config).to('cpu')
        model.eval()

        # torch.compile for faster CPU inference (PyTorch 2.x)
        _compiled = False
        try:
            model = torch.compile(model, mode='reduce-overhead', backend='inductor',
                                  fullgraph=False)
            _compiled = True
        except Exception as e:
            print(f"[Actor {rank}] torch.compile not available ({e}), using eager mode.")

        # Initialize memory bank (empty initially)
        memory_bank = None
        if config.use_engram:
            memory_bank = MemoryBank(config.memory_capacity, config.hidden_state_dim, config.hidden_state_dim)

        ipc_client = None
        if getattr(config, 'use_ipc_inference', False):
            ipc_client = InferenceClient(endpoint="tcp://127.0.0.1:5556")
            print(f"[Actor {rank}] Using ZeroMQ IPC Inference Client (5556).")
        
        print(f"[Actor {rank}] Started on CPU (compiled={_compiled}). PID: {os.getpid()}")
        
        last_load_time = 0
        last_weights_ver = -1
        games_played = 0

        # Phase 6: KOTH Actor State
        frozen_nets = {}      # {pid: HelperNetwork}
        active_pid = 1        # Default
        last_frozen_load_time = 0
        frozen_incompatible_warned = set()  # log "incompatible" once per pid
        
        # League Manager (Read-Only access for Opponent list)
        league = LeagueManager(config) # Will load league.json
        league_prob = getattr(config, 'league_opponent_prob', 0.2)
        league_opp = None # Current league opponent
        league_net = None # Network for league opponent

        while not stop_event.is_set():
            # 1. Load weights — prefer shared memory, fall back to file
            try:
                loaded = False

                # Fast path: shared memory (zero I/O, non-blocking lock)
                if shared_model is not None and weights_version is not None:
                    ver = weights_version.value
                    if ver > last_weights_ver:
                        lock = weights_lock if weights_lock is not None else None
                        acquired = False
                        try:
                            if lock:
                                acquired = lock.acquire(timeout=0.05)  # non-blocking
                            else:
                                acquired = True
                            if acquired:
                                # Load into unwrapped model if torch.compile is active
                                target = getattr(model, '_orig_mod', model)
                                target.load_state_dict(shared_model.state_dict())
                                last_weights_ver = ver
                                loaded = True
                        except Exception as e:
                            _log.error("[Actor %s] Shared memory load error: %s", rank, e)
                        finally:
                            if lock and acquired:
                                lock.release()
                        # If lock was busy, skip this round (use existing weights)

                # Slow path: file-based (only if shared memory is unavailable)
                if not loaded and shared_model is None and os.path.exists(weights_path):
                    mtime = os.path.getmtime(weights_path)
                    if mtime > last_load_time:
                        try:
                            state = torch.load(weights_path, map_location='cpu', weights_only=False)
                            target = getattr(model, '_orig_mod', model)
                            if isinstance(state, dict) and 'model' in state:
                                target.load_state_dict(state['model'])
                            else:
                                target.load_state_dict(state)
                            last_load_time = mtime
                            loaded = True
                        except Exception:
                            pass  # File contention, retry next loop

                if memory_bank and os.path.exists(memory_path):
                    if os.path.getmtime(memory_path) > last_load_time:
                        try:
                            mem_state = torch.load(memory_path, map_location='cpu', weights_only=False)
                            memory_bank.load_state_dict(mem_state)
                        except Exception: pass
                
                # Phase 6: Load KOTH Frozen Models
                if config.koth_mode:
                    frozen_path = os.path.join(config.checkpoint_dir, 'frozen_models.pt')
                    if os.path.exists(frozen_path):
                        fm_mtime = os.path.getmtime(frozen_path)
                        if fm_mtime > last_frozen_load_time:
                            try:
                                payload = torch.load(frozen_path, map_location='cpu', weights_only=False)
                                # payload is {'models': {pid: state}, 'active_pid': int}
                                f_models = payload.get('models', {})
                                active_pid = payload.get('active_pid', 1)
                                
                                # Update frozen_nets (create if missing)
                                for pid, state in f_models.items():
                                    if pid not in frozen_nets:
                                        # Create new instance on CPU
                                        net = MuZeroNetwork(config).to('cpu')
                                        net.eval()
                                        frozen_nets[pid] = net
                                    
                                    target = getattr(frozen_nets[pid], '_orig_mod', frozen_nets[pid])
                                    load_state = _strip_compiled_prefix(state) if isinstance(state, dict) else state
                                    missing, unexpected = target.load_state_dict(load_state, strict=False)
                                    if missing:
                                        # Architecture/checkpoint mismatch: use current policy weights for this frozen net
                                        try:
                                            src = shared_model if shared_model is not None else getattr(model, '_orig_mod', model)
                                            target.load_state_dict(src.state_dict())
                                            if rank == 0 and pid not in frozen_incompatible_warned:
                                                frozen_incompatible_warned.add(pid)
                                                print(f"[Actor {rank}] Frozen P{pid}: checkpoint incompatible ({len(missing)} missing keys), using current policy weights.")
                                        except Exception as e2:
                                            _log.warning("[Actor %s] Frozen P%s fallback load failed: %s", rank, pid, e2)
                                
                                last_frozen_load_time = fm_mtime
                            except Exception as e:
                                _log.warning("[Actor %s] Failed to load frozen models: %s", rank, e)

            except Exception as e:
                _log.error("[Actor %s] Error checking/loading weights: %s", rank, e)

            # 2. Play Session (multiple games with cumulative scoring)
            try:
                prev_total_games = shared_stats.total_games.value if rank == 0 else 0
                # Live View for Actor 0
                broadcast_fn = None
                if rank == 0 and live_queue is not None:
                    def broadcast_fn(event, data):
                        try:
                            live_queue.put({'event': event, 'data': data, 'time': time.time()})
                        except Exception: pass

                # Adaptive session length based on approximate training progress
                approx_step = shared_stats.total_steps.value
                session_len = get_adaptive_session_length(approx_step, config)

                # Curriculum settings
                current_curriculum_stage = 0
                if shared_curriculum_stage is not None:
                    current_curriculum_stage = shared_curriculum_stage.value
                elif 'state' in locals() and isinstance(state, dict):
                    current_curriculum_stage = state.get('stage', 0)
                
                if getattr(config, 'auto_curriculum', False):
                    # Ideally we init this once outside loop, but config might not have changed.
                    stage_params = CurriculumManager(config).stages[min(current_curriculum_stage, 3)]
                    board_size, win_length = stage_params.board_size, stage_params.win_length
                else:
                    board_size, win_length = config.board_size, config.win_length

                # Determine network argument
                network_arg = ipc_client if ipc_client is not None else model
                is_league_session = False
                current_opp_info = {}

                # League Check (Probabilistic)
                # But only if NOT KOTH and NOT PBT (Simplicity first)
                if not config.koth_mode and np.random.rand() < league_prob:
                    # Reload league data occasionally to see new opponents
                    # Since we play session, doing it per session is fine.
                    # We re-init league manager? Or just reload?
                    # LeagueManager.load() is fast (json read)
                    league.load()
                    opp = league.get_opponent()
                    
                    if opp and os.path.exists(opp.checkpoint_path):
                        # Load opponent
                        try:
                            if league_net is None:
                                league_net = MuZeroNetwork(config).to('cpu')
                                league_net.eval()
                            
                            # Load weights
                            ckpt = torch.load(opp.checkpoint_path, map_location='cpu', weights_only=False)
                            opp_state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
                            # Strip prefix if needed (though usually saved stripped)
                            opp_state = _strip_compiled_prefix(opp_state)
                            
                            # Handle mismatch
                            target = getattr(league_net, '_orig_mod', league_net)
                            # Filter keys
                            curr_state = target.state_dict()
                            filtered = {k: v for k, v in opp_state.items() if k in curr_state and v.shape == curr_state[k].shape}
                            target.load_state_dict(filtered, strict=False)
                            
                            # Setup 1 vs League
                            # Assuming 3 players. PID 1 = Current. PID 2,3 = League.
                            network_arg = {1: model}
                            for p in range(2, config.num_players + 1):
                                network_arg[p] = league_net
                                
                            is_league_session = True
                            league_opp = opp
                            current_opp_info = {p: {'elo': opp.elo, 'step': opp.step} for p in range(2, config.num_players + 1)}
                            # print(f"[Actor {rank}] Playing League Session vs Elo {opp.elo:.0f}")
                        except Exception as e:
                            _log.warning("[Actor %s] Failed to load league opp: %s", rank, e)
                            network_arg = model # Fallback
                    
                if not is_league_session and config.koth_mode:
                    network_arg = {}
                    base_net = ipc_client if ipc_client is not None else model
                    for pid in [1, 2, 3]:
                        if pid == active_pid:
                            network_arg[pid] = base_net
                        else:
                            network_arg[pid] = frozen_nets.get(pid, base_net)

                session_games = play_session(
                    network_arg, config,
                    session_length=session_len,
                    broadcast_fn=broadcast_fn,
                    game_index_base=rank * 100000 + games_played,
                    training_step=approx_step,
                    board_size_override=board_size,
                    win_length_override=win_length,
                    pbt_elo=league.current_elo
                )

                # 3. Send each game to queue (timeout to prevent deadlock if learner crashes)
                for game in session_games:
                    put_fail_count = 0
                    while not stop_event.is_set():
                        try:
                            if is_league_session:
                                game.is_evaluation = True
                                game.opponent_info = current_opp_info
                            game_queue.put(game, timeout=5.0)
                            break
                        except queue.Full:
                            put_fail_count += 1
                            if put_fail_count >= 10:
                                _log_learner.warning(
                                    "game_queue.put repeatedly full (%s times); learner may be slow.",
                                    put_fail_count,
                                )
                                put_fail_count = 0
                            continue
                    if stop_event.is_set():
                        break
                    shared_stats.update_game(game.winner, len(game), rankings=game.rankings)
                    games_played += 1

                # Update round stats (session = round in BO5)
                if session_games:
                    last_g = session_games[-1]
                    rw = getattr(last_g, 'round_winner', None)
                    rr = getattr(last_g, 'round_rankings', None)
                    if rr:
                        shared_stats.update_round(rw, rr)

                # Every 100 total games: save last game board image (Actor 0 only)
                # Use next available number from existing img/game_*.png so breakpoint resume never overwrites.
                if rank == 0 and session_games:
                    new_total = shared_stats.total_games.value
                    if new_total >= 100 and new_total // 100 > prev_total_games // 100:
                        last_game = session_games[-1]
                        board = getattr(last_game, 'final_board', None)
                        if board is not None:
                            img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'img')
                            os.makedirs(img_dir, exist_ok=True)
                            existing = []
                            try:
                                for f in os.listdir(img_dir):
                                    m = re.match(r'game_(\d+)\.png', f)
                                    if m:
                                        existing.append(int(m.group(1)))
                            except OSError:
                                pass
                            next_milestone = (max(existing) + 100) if existing else 100
                            path = os.path.join(img_dir, f'game_{next_milestone}.png')
                            if board_to_image_path(board, path):
                                print(f"[Actor 0] Saved board image: {path}", flush=True)

                if games_played % 10 == 0:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] [Actor {rank}] Finished game {games_played} (session={session_len}). Len: {len(session_games[-1])}", flush=True)
                
            except Exception as e:
                err_msg = str(e)
                if "NaN" in err_msg or "ModelCorrupted" in err_msg:
                    _log.error("[Actor %s] CRITICAL: Model corruption/NaN detected (%s). Forcing weight reload.", rank, err_msg)
                    last_load_time = 0 # Force reload next loop
                    time.sleep(1.0) # Brief backoff
                else:
                    _log.error("[Actor %s] Error playing game: %s", rank, e)
                    traceback.print_exc()
                
    except Exception as e:
        _log.error("[Actor %s] Crash: %s", rank, e)
        traceback.print_exc()


def learner_loop(config, args, game_queue, weights_path, memory_path, stop_event, shared_stats,
                 live_queue=None, shared_model=None, weights_version=None, weights_lock=None, shared_curriculum_stage=None):
    """Learner process: runs on GPU, trains network."""
    # Loop: (1) consume games from queue, (2) buffer and validate, (3) prefetch batch, (4) train_step, (5) checkpoint/curriculum/league/PBT, (6) optionally push weights to shared memory.
    try:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        print(f"[Learner] Started on {device}. PID: {os.getpid()}")
        
        # Initialize file logging (writes to checkpoint_dir/training.log)
        from ai.log_utils import setup_file_logging
        log_path = setup_file_logging(config.checkpoint_dir)
        print(f"[Learner] Training log: {log_path}")
        _log_learner.info("=== Training session started === Device: %s, PID: %d", device, os.getpid())
        _log_learner.info("Config: steps=%d, batch_size=%d, lr=%.2e, actors=%d, board=%d, num_simulations=%d",
                          args.steps, getattr(config, 'batch_size_start', 128), config.learning_rate,
                          args.actors, config.board_size, getattr(config, 'num_simulations_start', 25))
        
        # Start Dashboard
        start_ws_server(args.ws_port)
        
        # Init Model (compile happens AFTER resume/init to avoid _orig_mod prefix issues)
        model = MuZeroNetwork(config).to(device)
        model.train()

        # IPC Batched Inference Server
        if getattr(config, 'use_ipc_inference', False):
            _log.info("[Learner] Starting IPC Inference Server on GPU (5556)")
            ipc_server = InferenceServer(model, config, device, endpoint="tcp://127.0.0.1:5556")
            ipc_thread = threading.Thread(target=ipc_server.serve_forever, daemon=True)
            ipc_thread.start()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # Disable fp16 GradScaler: model is small enough for fp32 (4.5GB VRAM at bs=512)
        # fp16 autocast caused NaN from intermediate overflow (loss ~10 * scale=256 in fp16)
        scaler = torch.amp.GradScaler('cuda', enabled=False)

        # Learning rate scheduler: cosine annealing with linear warmup
        warmup_steps = 1000
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup: 0 → 1 over warmup_steps
                return max(0.01, current_step / warmup_steps)
            else:
                # Cosine annealing: 1 → min_lr_ratio over remaining steps
                progress = (current_step - warmup_steps) / max(1, args.steps - warmup_steps)
                return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Engram
        memory_bank = None
        if config.use_engram:
            memory_bank = MemoryBank(config.memory_capacity, config.hidden_state_dim, config.hidden_state_dim)

        # Replay Buffer Process Setup
        buffer_queue = mp.Queue(maxsize=1000)
        batch_queue = mp.Queue(maxsize=16)
        buffer_cmd_queue = mp.Queue()
        buffer_res_queue = mp.Queue()
        shared_buffer_games = mp.Value('i', 0)
        
        reanalyze_in_queue = mp.Queue(maxsize=100)
        reanalyze_out_queue = mp.Queue(maxsize=100)
        
        # Start buffer process
        p_buffer = mp.Process(target=buffer_loop, args=(
            config, buffer_queue, batch_queue, buffer_cmd_queue, buffer_res_queue, stop_event, shared_buffer_games,
            shared_stats, reanalyze_in_queue, reanalyze_out_queue
        ))
        p_buffer.start()
        
        # Start Reanalyze Worker
        p_reanalyze = mp.Process(target=reanalyze_worker, args=(
            0, config, reanalyze_in_queue, reanalyze_out_queue, stop_event, shared_model, weights_version, weights_lock
        ))
        p_reanalyze.start()
        
        def buffer_cmd_sync(cmd_dict, timeout=600):
            buffer_cmd_queue.put(cmd_dict)
            try:
                return buffer_res_queue.get(timeout=timeout)
            except Exception as e:
                _log.error(f"Buffer cmd {cmd_dict} failed/timeout: {e}")
                return None

        step = 0
        game_receive_count = 0
        latest_ckpt = os.path.join(config.checkpoint_dir, "latest.pt")

        # Phase 6: KOTH State
        koth_active_pid = 1  # Start with Red
        koth_step_counter = 0
        frozen_models = {}   # {pid: state_dict}

        # Phase 7: PBT Population
        population = None
        pbt_step_counter = 0
        active_agent_idx = 0
        
        if config.pbt_population_size > 1:
             print(f"[Learner] PBT Enabled. Population size: {config.pbt_population_size}")
             population = Population(config)

        # Phase 3: Curriculum Manager & League
        curriculum = CurriculumManager(config)
        league = LeagueManager(config)
        
        # Load initial stage params from Curriculum (if auto)
        if getattr(config, 'auto_curriculum', False):
            # If resume, it loads stage index later.
            pass

        last_curriculum_check_games = 0
        last_curriculum_check_wins = 0
        board_size_discard_count = 0  # Track games discarded due to curriculum stage mismatch

        # ── Resume from checkpoint ──────────────────────────────────
        data = None
        step = 0
        model_loaded_from_ckpt = False  # True if model was loaded from latest.pt (full or weights-only)
        if args.resume:
            try:
                # 1. Load model / optimizer / scaler / stats from latest.pt
                if os.path.exists(latest_ckpt):
                    print(f"[Learner] Resuming from {latest_ckpt}...")
                    data = torch.load(latest_ckpt, map_location=device, weights_only=False)
                if isinstance(data, dict) and 'model' in data:
                    ckpt_state = data['model']
                    # Drop old flat policy output keys (incompatible with factored row+col)
                    drop_keys = [k for k in ckpt_state if k.startswith('prediction.policy_net.6.')
                                 or k.startswith('prediction.policy_out.')]
                    for k in drop_keys:
                        del ckpt_state[k]
                        print(f"[Learner] Dropped old checkpoint key: {k}")
                    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
                    if missing:
                        print(f"[Learner] New params initialized (architecture expansion): {len(missing)} keys")
                        for k in missing:
                            print(f"  + {k}")
                    if unexpected:
                        print(f"[Learner] Ignored old keys: {len(unexpected)} keys")
                    try:
                        # Cross-compatible: try 'optimizer' then 'optimizer_state_dict'
                        opt_state = data.get('optimizer', data.get('optimizer_state_dict'))
                        if opt_state:
                            optimizer.load_state_dict(opt_state)
                    except ValueError as e:
                        print(f"[Learner] Optimizer state incompatible (architecture changed), resetting: {e}")
                    # Skip loading scaler state — we now use fp32 only (scaler disabled)
                    # Cross-compatible step: try 'step' then 'global_step'
                    step = data.get('step', data.get('global_step', 0))
                    model_loaded_from_ckpt = True
                else:
                    if data is not None:
                        print(f"[Learner] Loaded weights-only checkpoint from {latest_ckpt}")
                        model.load_state_dict(data, strict=False)
                        step = 0 # No step info in weights-only
                        model_loaded_from_ckpt = True
                    else:
                        print(f"[Learner] No checkpoint found at {latest_ckpt}. Starting with fresh weights.")
                        step = 0
                
                shared_stats.total_steps.value = step

                # Restore stats (strict: always prefer checkpoint; fallback for old/weights-only format)
                stats = data.get('stats', {}) if isinstance(data, dict) else {}
                # Cross-compat: train.py uses top-level 'total_games'; ensure we never reset to 0 when ckpt has it
                ckpt_games = stats.get('games') if isinstance(stats, dict) else None
                if ckpt_games is None and isinstance(data, dict):
                    ckpt_games = data.get('total_games', 0)
                ckpt_games = ckpt_games if ckpt_games is not None else 0
                with shared_stats.lock:
                    shared_stats.total_games.value = int(ckpt_games)
                    shared_stats.total_game_len.value = int(stats.get('total_len', 0)) if isinstance(stats, dict) else 0
                    wc = stats.get('wins', {}) if isinstance(stats, dict) else {}
                    shared_stats.win_counts[0] = wc.get('1', 0)
                    shared_stats.win_counts[1] = wc.get('2', 0)
                    shared_stats.win_counts[2] = wc.get('3', 0)
                    shared_stats.win_counts[3] = wc.get('draw', 0)
                    shared_stats.ranked_games.value = int(stats.get('ranked_games', 0)) if isinstance(stats, dict) else 0
                    plc = stats.get('placements', {}) if isinstance(stats, dict) else {}
                    for pid in (1, 2, 3):
                        arr = plc.get(str(pid), [0, 0, 0])
                        for pl in range(3):
                            shared_stats.placement_counts[(pid - 1) * 3 + pl] = arr[pl] if pl < len(arr) else 0
                    # Round-level stats (Best-of-5)
                    if isinstance(stats, dict):
                        shared_stats.total_rounds.value = int(stats.get('total_rounds', 0))
                        rw = stats.get('round_wins', {})
                        for i in range(4):
                            key = 'draw' if i == 3 else str(i + 1)
                            shared_stats.round_win_counts[i] = int(rw.get(key, 0))
                        rpc = stats.get('round_placements', {})
                        for pid in (1, 2, 3):
                            arr = rpc.get(str(pid), [0, 0, 0])
                            for pl in range(3):
                                shared_stats.round_placement_counts[(pid - 1) * 3 + pl] = arr[pl] if pl < len(arr) else 0
                
                # Checkpoint KOTH restore
                koth_active_pid = data.get('koth_active_pid', 1) if isinstance(data, dict) else 1
                koth_step_counter = data.get('koth_step_counter', 0) if isinstance(data, dict) else 0
                frozen_models = data.get('frozen_models', {}) if isinstance(data, dict) else {}
                if config.koth_mode:
                    print(f"[Learner] KOTH Restore: Active=P{koth_active_pid}, Step={koth_step_counter}, Frozen={list(frozen_models.keys())}")

                # Checkpoint PBT restore
                if population and isinstance(data, dict) and 'population' in data:
                     # New way: PBT state embedded in latest.pt?
                     pass
                
                # Check for separate population file
                if population:
                     pop_path = os.path.join(config.checkpoint_dir, 'population.pt')
                     if os.path.exists(pop_path):
                         population.load(pop_path)
                         # Restore active agent state
                         active_agent_idx = data.get('pbt_active_agent_idx', 0) if isinstance(data, dict) else 0
                         pbt_step_counter = data.get('pbt_step_counter', 0) if isinstance(data, dict) else 0
                         print(f"[Learner] PBT Restore: Gen {population.generation}, Active Agent {active_agent_idx}")
                         
                         # Load active agent weights into model/optimizer (overriding latest.pt to ensure consistency)
                         population.load_agent_weights(active_agent_idx, model, optimizer)
                         print(f"[Learner] Loaded Active Agent {active_agent_idx} weights.")
                     else:
                         # population.pt is only created on fresh start (no --resume) or after PBT evolution.
                         # Missing when: first time using PBT with --resume, or checkpoint dir had no PBT run before.
                         print("[Learner] Population checkpoint missing (normal if resuming from non-PBT or first PBT resume). Initializing from current model.")
                         for i in range(population.size):
                             population.sync_agent_weights(i, model, optimizer)
                         population.save(os.path.join(config.checkpoint_dir, 'population.pt'))
                         print("[Learner] Saved population.pt for future resume.")

                # Checkpoint Curriculum & League restore
                if curriculum and isinstance(data, dict) and 'curriculum_state' in data:
                    curriculum.load_state_dict(data['curriculum_state'])
                    st = curriculum.get_current_stage()
                    print(f"[Learner] Curriculum Restore: Stage {st.stage_id} ({st.board_size}x{st.board_size})")
                
                if league and isinstance(data, dict) and 'league_current_elo' in data:
                    league.current_elo = data['league_current_elo']
                    print(f"[Learner] League Restore: Elo {league.current_elo}")

                print(f"[Learner] Resumed at Step {step}, Games {shared_stats.total_games.value}")

                # ── Sanitize corrupted state ─────────────────────────
                # Fix GradScaler scale=0 or extremely small (causes div-by-zero → NaN gradients)
                scaler_state = scaler.state_dict()
                if scaler_state.get('scale', 1.0) <= 0:
                    print(f"[Learner] WARNING: GradScaler scale={scaler_state['scale']}. Resetting to safe value (1024).")
                    scaler = torch.amp.GradScaler('cuda', enabled=False)

                # Fix NaN/Inf in model weights & buffers (e.g. BatchNorm running stats)
                nan_fixed = 0
                for name, param in list(model.named_parameters()) + list(model.named_buffers()):
                    is_var = 'running_var' in name or 'var' in name
                    
                    if torch.isnan(param).any() or torch.isinf(param).any() or (is_var and (param < 0).any()):
                        with torch.no_grad():
                            # If it's a variance (running_var), reset to 1.0
                            if is_var:
                                param[torch.isnan(param)] = 1.0
                                param[torch.isinf(param)] = 1.0
                                param[param < 0] = 1.0
                            else:
                                param[torch.isnan(param)] = 0.0
                                param[torch.isinf(param)] = 0.0
                        nan_fixed += 1
                        print(f"[Learner] WARNING: Fixed NaN/Inf/NegVar in {name}")
                
                if nan_fixed or scaler_state.get('scale', 1.0) <= 0:
                    print(f"[Learner] Sanitized {nan_fixed} tensors. Resetting optimizer state.")
                    if nan_fixed > 0:
                        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                        for param_group in optimizer.param_groups:
                            param_group.setdefault('initial_lr', config.learning_rate)
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=step - 1)
                        # Save the fixed FULL checkpoint (not just state_dict!) so resume still works
                        # Use current shared_stats so we never persist empty stats (e.g. after old-format load)
                        fixed_data = {
                            'step': step,
                            'model': _strip_compiled_prefix(model.state_dict()),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'stats': shared_stats.get_info(),
                            'curriculum_state': curriculum.state_dict(),
                            'league_current_elo': league.current_elo
                        }
                        atomic_torch_save(fixed_data, latest_ckpt)
                        print(f"[Learner] Saved sanitized checkpoint to {latest_ckpt}")

                # Fallback: only when we never loaded from latest.pt (e.g. file missing), try shared_weights.pt
                if not model_loaded_from_ckpt and os.path.exists(weights_path):
                    print(f"[Learner] latest.pt missing, loading weights only from {weights_path}...")
                    try:
                        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
                        print("[Learner] Weights loaded. (Optimizer/Step reset)")
                    except Exception as e:
                        _log_learner.warning("Failed to load weights: %s. Try without --resume or check file.", e)

                # 2. Load Memory Bank (shared_memory.pt)
                if memory_bank and os.path.exists(memory_path):
                    print(f"[Learner] Loading memory bank from {memory_path}...")
                    try:
                        mem_state = torch.load(memory_path, map_location='cpu', weights_only=False)
                        print(f"[Learner] Memory state keys: {mem_state.keys()}, count={mem_state.get('count', '?')}")
                        memory_bank.load_state_dict(mem_state)
                        print(f"[Learner] Memory bank loaded. Size: {memory_bank.count}")
                    except Exception as e:
                        _log.warning("[Learner] Failed to load memory bank: %s", e)

                # 3. Load Replay Buffer (replay_buffer.pkl)
                buffer_path = os.path.join(config.checkpoint_dir, 'replay_buffer.pkl')
                if os.path.exists(buffer_path) or os.path.exists(buffer_path + "_chunks") or os.path.exists(buffer_path + ".tmp"):
                    print(f"[Learner] Telling Buffer Process to load replay buffer from {buffer_path}...")
                    res = buffer_cmd_sync({'type': 'load', 'path': buffer_path})
                    if res and res.get('type') == 'load_ok':
                        print(f"[Learner] Replay buffer loaded. Games: {res.get('num_games')}")
                    else:
                        print("[Learner] Buffer Process failed to load replay buffer.")
                else:
                    print(f"[Learner] Replay buffer not found. Starting with empty buffer.")

                # 4. Load Metrics Log
                metrics_path = os.path.join(config.checkpoint_dir, 'metrics_log.jsonl')
                try:
                    ms = load_metrics_log(metrics_path)
                    print(f"[Learner] Loaded {len(ms)} metrics history entries.")
                except Exception as e:
                    _log.warning("[Learner] Failed to load metrics log: %s", e)

                # 5. Fast-forward scheduler to current step (use last_epoch to avoid step() before optimizer.step() warning)
                if step > 0:
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=step - 1)
                    print(f"[Learner] LR scheduler fast-forwarded to step {step}, lr={optimizer.param_groups[0]['lr']:.2e}")

                # 6. Initialize Curriculum Stage (for auto-curriculum)
                loaded_stage = data.get('curriculum_stage', -1) if (data is not None and isinstance(data, dict)) else -1

                # 7. Now sync the LOADED state to shared files so actors pick it up
                atomic_torch_save(_strip_compiled_prefix(model.state_dict()), weights_path)
                if memory_bank and memory_bank.count > 0:
                    atomic_torch_save(memory_bank.state_dict(), memory_path)
                    print(f"[Learner] Synced resumed weights + memory ({memory_bank.count} entries) to shared files.")
                else:
                    print("[Learner] Synced resumed weights to shared files. (Memory empty or not loaded, NOT overwriting disk.)")

            except Exception as e:
                _log_learner.error("Resume failed: %s. Start without checkpoint or fix file.", e)
                _log.error("[Learner] Resume failed. Use --no-resume or check checkpoint file.")
                data = None
                step = 0

        if args.resume:
            pass # (Resume block above)
        else:
            # ── Fresh start: save initial empty state for actors ────
            atomic_torch_save(_strip_compiled_prefix(model.state_dict()), weights_path)
            if memory_bank:
                atomic_torch_save(memory_bank.state_dict(), memory_path)
            print("[Learner] Fresh start. Initial weights saved for actors.")
            
            # PBT Fresh Start
            if population:
                print("[Learner] Initializing PBT population from initial model...")
                for i in range(population.size):
                    population.sync_agent_weights(i, model, optimizer)
                population.save(os.path.join(config.checkpoint_dir, 'population.pt'))

        # Phase 6: KOTH compatibility check — disk file and in-memory (so Actors never see incompatible state)
        frozen_path = os.path.join(config.checkpoint_dir, 'frozen_models.pt')
        if config.koth_mode:
            ref_keys = set(_strip_compiled_prefix(model.state_dict()).keys())
            # 1) If file on disk exists, check it (covers no-resume or checkpoint had no frozen_models)
            if os.path.exists(frozen_path):
                try:
                    payload = torch.load(frozen_path, map_location='cpu', weights_only=False)
                    f_models = payload.get('models', {})
                    sample = next((f_models[pid] for pid in [1, 2, 3] if pid in f_models), None)
                    if sample is not None:
                        have_keys = set(sample.keys())
                        if ref_keys - have_keys:
                            print(f"[Learner] KOTH frozen_models.pt on disk incompatible ({len(ref_keys - have_keys)} missing keys), removing and re-initializing.")
                            try:
                                os.remove(frozen_path)
                            except OSError:
                                pass
                            frozen_models = {}
                except Exception:
                    pass
            # 2) If in-memory frozen_models from checkpoint, check and clear if incompatible
            if frozen_models:
                sample = next((frozen_models[pid] for pid in [1, 2, 3] if pid in frozen_models), None)
                if sample is not None:
                    have_keys = set(sample.keys())
                    if ref_keys - have_keys:
                        print(f"[Learner] KOTH frozen_models from checkpoint incompatible ({len(ref_keys - have_keys)} missing keys), re-initializing.")
                        frozen_models = {}
                        if os.path.exists(frozen_path):
                            try:
                                os.remove(frozen_path)
                            except OSError:
                                pass

        # Phase 6: KOTH Initialization (runs for both Resume and Fresh Start)
        if config.koth_mode and not frozen_models:
             print("[Learner] KOTH enabled but no frozen models found. Initializing independent populations from current weights.")
             # Prepare dict
             current_sd = _strip_compiled_prefix(model.state_dict())
             ref_keys = set(current_sd.keys())
             
             if population:
                 # If PBT, use population agents as frozen models
                 # Just drop in whatever agent 1, 2, 3 are.
                 # If pop size > 3, we just use first 3 for KOTH slots initially.
                 # If pop size < 3, we cycle.
                 for pid in [1, 2, 3]:
                    agent_idx = (pid - 1) % population.size
                    # Load that agent's model state
                    # Note: we need to be careful about device. Population stores on CPU.
                    state = _strip_compiled_prefix(population.agents[agent_idx].model_state)
                    frozen_models[pid] = _ensure_frozen_state_compatible(state, ref_keys, current_sd, f"P{pid} from PBT Agent {agent_idx}")
                    print(f"  - Initialized KOTH Player {pid} from PBT Agent {agent_idx}")
             else:
                 # Standard KOTH (no PBT)
                 for pid in [1, 2, 3]:
                     frozen_models[pid] = {k: v.clone() for k, v in current_sd.items()}
             
             # Reset state
             koth_active_pid = 1 
             koth_step_counter = 0

             # Save immediately so actors pick it up
             _sanitize_frozen_models_for_save(frozen_models, model)
             frozen_path = os.path.join(config.checkpoint_dir, 'frozen_models.pt')
             koth_state_payload = {'models': frozen_models, 'active_pid': koth_active_pid}
             atomic_torch_save(koth_state_payload, frozen_path)
             print(f"[Learner] Saved initial frozen models to {frozen_path}")

        # Sync initial weights to shared memory so actors can pick them up immediately
        if shared_model is not None and weights_version is not None:
            try:
                cpu_state = _strip_compiled_prefix(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
                lock = weights_lock if weights_lock is not None else None
                if lock:
                    lock.acquire()
                try:
                    shared_model.load_state_dict(cpu_state)
                    weights_version.value += 1
                finally:
                    if lock:
                        lock.release()
                print("[Learner] Initial weights synced to shared memory.")
            except Exception as e:
                _log.error("[Learner] Shared memory init sync failed: %s", e)

        # torch.compile for faster GPU training (PyTorch 2.x)
        # Done AFTER resume so load_state_dict works with clean key names.
        # After this, any load_state_dict into model must use getattr(model, '_orig_mod', model) or state_dict with keys already stripped.
        _learner_compiled = False
        try:
            model = torch.compile(model, mode='default', backend='inductor',
                                  fullgraph=False)
            _learner_compiled = True
            print(f"[Learner] Model compiled with torch.compile (inductor).")
        except Exception as e:
            print(f"[Learner] torch.compile not available ({e}), using eager mode.")

        t_start = time.time()
        nan_streak = 0  # Consecutive NaN loss counter

        # Async batch prefetcher: pulls from batch_queue and pins tensors in a pipeline
        from collections import deque
        prefetcher = ThreadPoolExecutor(max_workers=config.prefetch_workers)

        def _prefetch_batch():
            """Fetch a pre-sampled batch from background process and pin to GPU."""
            raw = batch_queue.get()
            pinned = {}
            for key, val in raw.items():
                t = torch.from_numpy(val)
                if device.type == 'cuda':
                    t = t.pin_memory()
                pinned[key] = t
            return pinned
            
        prefetch_futures = deque()
        for _ in range(3): # Pipeline depth of 3
            prefetch_futures.append(prefetcher.submit(_prefetch_batch))

        last_stage = -1
        
        # Unconditionally restore curriculum and league states from checkpoint
        if data is not None and isinstance(data, dict):
            if 'curriculum_state' in data:
                curriculum.load_state_dict(data['curriculum_state'])
                last_stage = curriculum.current_stage_idx
            elif 'curriculum_stage' in data:
                raw = int(data['curriculum_stage'])
                last_stage = (raw - 1) if 1 <= raw <= 4 else raw
                last_stage = min(len(curriculum.stages) - 1, max(0, last_stage))
                curriculum.set_stage(last_stage)
            else:
                _log_learner.warning("Resume: no curriculum_state or curriculum_stage in checkpoint; using default stage.")

            if 'league_current_elo' in data:
                league.current_elo = data['league_current_elo']
                _log_learner.info("League Elo restored: %.1f", league.current_elo)
            else:
                _log_learner.warning("Resume: league_current_elo missing in checkpoint; keeping default 1200.")
                
        if getattr(config, 'auto_curriculum', False):
            # If resume didn't specify or it's a fresh start, calculate from step.
            if last_stage == -1:
                 # Fallback / Fresh start
                 last_stage = curriculum.current_stage_idx
                 
            if shared_curriculum_stage is not None:
                 shared_curriculum_stage.value = last_stage

        while step < args.steps and not stop_event.is_set():
            # 0. Curriculum Check (Learner side)
            if getattr(config, 'auto_curriculum', False):
                # Ensure local var tracks manager
                if curriculum.current_stage_idx > last_stage:
                    print(f"[Learner] Curriculum Advancement: Stage {last_stage} -> {curriculum.current_stage_idx}", flush=True)
                    print(f"[Learner] Clearing Replay Buffer to prevent stage mismatch...", flush=True)
                    buffer_cmd_sync({'type': 'clear'})
                    last_stage = curriculum.current_stage_idx

            # 1. Empty Queue
            fetched = 0
            
            # Determine expected board size for current stage
            expected_size = config.board_size
            if getattr(config, 'auto_curriculum', False):
                st = curriculum.get_current_stage()
                expected_size = st.board_size

            while not game_queue.empty() and fetched < 200:
                try:
                    game = game_queue.get_nowait()
                    
                    # Verify game matches current curriculum stage
                    if game.board_size != expected_size:
                        board_size_discard_count += 1
                        if board_size_discard_count <= 5 or board_size_discard_count % 50 == 0:
                            print(f"[Learner] Discarded game: board_size={game.board_size} != expected {expected_size} (total discarded: {board_size_discard_count})", flush=True)
                        continue

                    # Score for curriculum: 1.0 win, 0.5 draw, 0.0 loss (for current/active agent)
                    winner = game.winner
                    if getattr(game, 'is_evaluation', False):
                        # League game: PID 1 = current agent
                        score = 0.5
                        if winner == 1: score = 1.0
                        elif winner is None: score = 0.5
                        else: score = 0.0
                    else:
                        # KOTH game: koth_active_pid = current agent
                        score = 0.5
                        if winner == koth_active_pid: score = 1.0
                        elif winner is None: score = 0.5
                        else: score = 0.0
                    curriculum.record_game_result(score)

                    # League-only: update Elo and opponent
                    if getattr(game, 'is_evaluation', False):
                        opp_info = getattr(game, 'opponent_info', {})
                        if opp_info:
                            from ai.league import LeagueOpponent
                            if 2 in opp_info:
                                o_dat = opp_info[2]
                                op_obj = LeagueOpponent("", o_dat['elo'], o_dat['step'], 0)
                                league.record_match(op_obj, score)

                    # Offload game storage to ReplayBuffer process
                    buffer_queue.put((game, step))
                    game_receive_count += 1
                    
                    if memory_bank:
                        from ai.train import _update_memory_bank
                        _update_memory_bank(memory_bank, game, model, device, config)
                        
                    fetched += 1
                except Exception:
                    break
            
            # Broadcast reception
            if fetched > 0:
                stats = shared_stats.get_info()
                broadcast('batch_stats', { 
                    'total_games': stats['games'],
                    'win_counts': stats['wins'],
                    'ranked_games': stats['ranked_games'],
                    'placements': stats['placements'],
                })

            # Handle League Results comment (deprecated, logic moved inside loop)



            # 2. Train
            if shared_buffer_games.value < config.min_buffer_size:
                # Discard any pending prefetch — buffer state may have changed
                pending_batch_future = None

                if 'start_wait_time' not in locals():
                    start_wait_time = time.time()
                
                time.sleep(0.2)
                elapsed = time.time() - start_wait_time
                count = shared_buffer_games.value
                if count > 0 and elapsed > 1.0:
                    rate = count / elapsed  # games/sec
                    remaining = config.min_buffer_size - count
                    eta_sec = remaining / rate
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                    speed_str = f"{rate * 60:.1f} games/min"
                else:
                    eta_str = "Calculating..."
                    speed_str = "..."

                # Throttle prints to every 2s; dashboard update every loop for responsiveness
                if 'last_data_gen_print' not in locals() or (time.time() - last_data_gen_print) >= 2.0:
                    last_data_gen_print = time.time()
                    current_time = datetime.now().strftime("%H:%M:%S")
                    percent = (count / config.min_buffer_size) * 100
                    recv_hint = f" (received {game_receive_count})" if game_receive_count > 0 and count == 0 else ""
                    if count == 0 and board_size_discard_count > 0:
                        recv_hint += f" (discarded {board_size_discard_count} games: board_size mismatch, expect {expected_size}x{expected_size})"
                    wait_hint = " (30x30 games ~1-2 min each; actors in progress)" if count == 0 and elapsed > 30 else ""
                    print(f"[{current_time}] [Data Gen] Progress: {count}/{config.min_buffer_size} ({percent:.1f}%) | Speed: {speed_str} | ETA: {eta_str}{recv_hint}{wait_hint}", flush=True)
                
                stats = shared_stats.get_info()
                broadcast('status', {
                    'total_games': stats['games'],
                    'win_counts': stats['wins'],
                    'ranked_games': stats['ranked_games'],
                    'placements': stats['placements'],
                    'total_rounds': stats['total_rounds'],
                    'round_wins': stats['round_wins'],
                    'round_placements': stats['round_placements'],
                    'buffer_games': count,
                    'step': f"Gen {count}/{config.min_buffer_size}",
                    'lr': 0.0
                })
                
                
                # Forward Live Events (while waiting)
                if live_queue:
                    while not live_queue.empty():
                        try:
                            evt = live_queue.get_nowait()
                            broadcast(evt['event'], evt['data'])
                        except Exception: break
                
                continue
                
            # ── Async prefetch pipeline with gradient accumulation (2 micro-steps) ──
            
            # Phase 7: PBT Evolution
            if population and config.pbt_population_size > 1:
                pbt_step_counter += 1
                
                # Track stats for PBT (Naive: use global winrate delta)
                # Ideally we track this specific agent's performance.
                # Since only ONE agent learns at a time, we can attribute recent performance to it.
                
                if pbt_step_counter >= config.pbt_period:
                     # 1. Update stats for current active agent
                     # We need to snapshot wins/games at start of period.
                     # For now, let's just use current global winrate as a proxy or 0.5.
                     # Better: use the shared_stats snapshot we take for logs?
                     # Let's implement proper tracking later. For now, assign random score or keep 0?
                     # Actually, let's assign the current 'win rate' of the active player (active_pid) from stats?
                     # But active_pid changes due to KOTH!
                     # So the agent plays as P1, then P2, then P3.
                     # We should track TOTAL wins of (P1+P2+P3) vs total games during this PBT period?
                     # Since the agent IS all 3 players (sequentially), its performance is the sum.
                     # Yes.
                     
                     # Sync weights to population
                     population.sync_agent_weights(active_agent_idx, model, optimizer)
                     
                     # Evolve
                     population.exploit_and_explore()
                     
                     # Rotate Active Agent
                     active_agent_idx = (active_agent_idx + 1) % population.size
                     print(f"[PBT] Step {step}: Evolving... Rotating to Agent {active_agent_idx}")
                     
                     # Load new agent
                     population.load_agent_weights(active_agent_idx, model, optimizer)
                     
                     # Reset PBT counters
                     pbt_step_counter = 0
                     
                     # Re-init KOTH state to be consistent with new agent context
                     koth_active_pid = 1
                     koth_step_counter = 0
                     frozen_models = {}
                     current_sd = _strip_compiled_prefix(model.state_dict())
                     ref_keys = set(current_sd.keys())
                     for pid in [1, 2, 3]:
                        a_idx = (pid - 1) % population.size
                        state = _strip_compiled_prefix(population.agents[a_idx].model_state)
                        frozen_models[pid] = _ensure_frozen_state_compatible(state, ref_keys, current_sd, f"P{pid} after PBT evolution")
                     
                     # Save state
                     _sanitize_frozen_models_for_save(frozen_models, model)
                     population.save(os.path.join(config.checkpoint_dir, 'population.pt'))
                     atomic_torch_save({'models': frozen_models, 'active_pid': koth_active_pid}, 
                                       os.path.join(config.checkpoint_dir, 'frozen_models.pt'))
                     
                     # Reset scheduler? 
                     # load_agent_weights restores optimizer state (including LR).
                     # But we might want to ensure scheduler matches?
                     # Scheduler wraps optimizer. If optimizer param groups change (LR), scheduler might overwrite them?
                     # LambdaLR sets LR based on step. 
                     # PBT modifies LR. 
                     # We need PBT to override Scheduler?
                     # Or Scheduler to respect PBT?
                     # If we use LambdaLR, it sets lr = initial_lr * lambda(epoch).
                     # PBT changes initial_lr? No, PBT changes current LR.
                     # If we want PBT to control LR, we should disable Scheduler or make it multiplicative?
                     # For now, let's assume PBT mutation updates the optimizer's LR, 
                     # and we reset the scheduler to base LR?
                     # Actually, if we use a scheduler, PBT mutation is fighting it.
                     # Let's re-init scheduler with the NEW base LR from the agent?
                     # agent.learning_rate is the "base" LR.
                     for param_group in optimizer.param_groups:
                         param_group['initial_lr'] = population.agents[active_agent_idx].learning_rate
                     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=step)

            # Phase 6: KOTH Rotation
            if config.koth_mode:
                koth_step_counter += 1
                if koth_step_counter >= config.koth_period:
                    # Rotate Active Player
                    old_pid = koth_active_pid
                    koth_active_pid = (koth_active_pid % 3) + 1
                    koth_step_counter = 0
                    print(f"[KOTH] Rotating Active Player: P{old_pid} -> P{koth_active_pid}")
                    
                    # 1. Snapshot old player weights
                    # Clone to avoid reference issues if model mutates in place
                    current_sd = _strip_compiled_prefix(model.state_dict())
                    frozen_models[old_pid] = {k: v.clone() for k, v in current_sd.items()}
                    
                    # 2. Load new player weights (into unwrapped model: frozen_models use stripped keys)
                    if koth_active_pid in frozen_models:
                        target = getattr(model, '_orig_mod', model)
                        missing, unexpected = target.load_state_dict(frozen_models[koth_active_pid], strict=False)
                        if missing:
                            print(f"[KOTH] WARNING: P{koth_active_pid} load had {len(missing)} missing keys (incompatible?), continuing.")
                            # Overwrite slot with full current model state so next rotation/Actors see compatible state
                            frozen_models[koth_active_pid] = {k: v.clone() for k, v in _strip_compiled_prefix(model.state_dict()).items()}
                            print(f"[KOTH] Replaced P{koth_active_pid} frozen slot with current model weights.")
                        if unexpected:
                            print(f"[KOTH] WARNING: P{koth_active_pid} load had {len(unexpected)} unexpected keys.")
                        # Reset optimizer momentum
                        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                        for param_group in optimizer.param_groups:
                            param_group.setdefault('initial_lr', config.learning_rate)
                        # Re-bind scheduler keeping global step
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=step)
                        print(f"[KOTH] Loaded P{koth_active_pid} weights. Optimizer reset.")

                    # 3. Save KOTH state for Actors (Wrapper dict to include active_pid)
                    _sanitize_frozen_models_for_save(frozen_models, model)
                    frozen_path = os.path.join(config.checkpoint_dir, 'frozen_models.pt')
                    koth_state_payload = {'models': frozen_models, 'active_pid': koth_active_pid}
                    atomic_torch_save(koth_state_payload, frozen_path)

            accum_steps = 2
            optimizer.zero_grad()  

            for micro in range(accum_steps):
                batch = prefetch_futures.popleft().result()
                prefetch_futures.append(prefetcher.submit(_prefetch_batch))

                is_last_micro = (micro == accum_steps - 1)
                try:
                    loss_dict = train_step(
                        model, optimizer, scaler, batch, config, device, memory_bank,
                        step_counter=step,
                        accumulate=not is_last_micro,
                        koth_active_pid=koth_active_pid if config.koth_mode else None,
                        accum_scale=1.0 / accum_steps
                    )
                    
                    # Ensure update priorities for PER
                    if not loss_dict.get('_nan', False) and 'td_errors' in loss_dict and 'insert_idxs' in batch:
                        try:
                            buffer_cmd_queue.put_nowait({
                                'type': 'update_priorities',
                                'indices': batch['insert_idxs'].cpu().numpy(),
                                'errors': loss_dict['td_errors']
                            })
                        except queue.Full:
                            pass
                except Exception as e:
                    _log.exception("[Learner] train_step exception: %s", e)
                    traceback.print_exc()
                    optimizer.zero_grad()  
                    loss_dict = {'total': float('nan'), '_nan': True}

            
            # ── NaN recovery logic (graduated 3-tier) ──
            if loss_dict.get('_nan', False):
                nan_streak += 1
                if nan_streak % 10 == 1:
                    _log.warning("[Learner] NaN loss detected (streak=%s). Skipping step.", nan_streak)
                
                # Tier 1 (10 NaN streak): Halve learning rate temporarily
                if nan_streak == 10:
                    _log.warning("[Learner] NaN Tier 1: Halving learning rate temporarily.")
                    for pg in optimizer.param_groups:
                        pg['lr'] = pg['lr'] * 0.5
                
                # Tier 2 (30 NaN streak): Sanitize model weights + reset scaler
                if nan_streak == 30:
                    _log.error("[Learner] NaN Tier 2: Sanitizing model weights and resetting scaler.")
                    fixed = 0
                    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            with torch.no_grad():
                                if 'running_var' in name or 'var' in name:
                                    param[torch.isnan(param) | torch.isinf(param)] = 1.0
                                else:
                                    param[torch.isnan(param) | torch.isinf(param)] = 0.0
                            fixed += 1
                    scaler = torch.amp.GradScaler('cuda', enabled=False)
                    print(f"[Learner] Tier 2: Sanitized {fixed} tensors, reset scaler.", flush=True)

                # Tier 3 (50 NaN streak): Full optimizer reset (last resort)
                if nan_streak >= 50:
                    _log.error("[Learner] NaN Tier 3 CRITICAL: %s consecutive NaN losses. Full optimizer reset.", nan_streak)
                    # Re-sanitize anything Tier 2 might have missed
                    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            with torch.no_grad():
                                if 'running_var' in name or 'var' in name:
                                    param[torch.isnan(param) | torch.isinf(param)] = 1.0
                                else:
                                    param[torch.isnan(param) | torch.isinf(param)] = 0.0
                    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                    for param_group in optimizer.param_groups:
                        param_group.setdefault('initial_lr', config.learning_rate)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=step - 1)
                    scaler = torch.amp.GradScaler('cuda', enabled=False)
                    print(f"[Learner] Tier 3: Full optimizer + scheduler + scaler reset.", flush=True)
                    nan_streak = 0
                
                continue  # Skip this step entirely (don't increment step counter)
            else:
                nan_streak = 0
            
            step += 1
            shared_stats.total_steps.value = step
            scheduler.step()
            
            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Learner] Step {step} | Loss: {loss_dict['total']:.4f} (F:{loss_dict.get('focus', 0):.4f}) | LR: {current_lr:.2e} | Games: {game_receive_count}", flush=True)

                # Forward Live Events (during training)
                if live_queue:
                    fetched_evts = 0
                    while not live_queue.empty() and fetched_evts < 50:
                        try:
                            evt = live_queue.get_nowait()
                            broadcast(evt['event'], evt['data'])
                            fetched_evts += 1
                        except Exception: break
                
                # Broadcast Metrics (include buffer_games so dashboard shows current buffer)
                stats_snapshot = shared_stats.get_info()
                m = {
                    'step': step,
                    'buffer_games': shared_buffer_games.value,
                    'loss': float(loss_dict['total']),
                    'loss_value': float(loss_dict.get('value', 0.0)),
                    'loss_reward': float(loss_dict.get('reward', 0.0)),
                    'loss_policy': float(loss_dict.get('policy', 0.0)),
                    'loss_focus': float(loss_dict.get('focus', 0.0)),
                    'loss_recon': float(loss_dict.get('recon', 0.0)),
                    'avg_game_length': float(stats_snapshot['avg_len']),
                    'lr': optimizer.param_groups[0]['lr'],
                    'total_games': stats_snapshot['games'],
                    'win_counts': stats_snapshot['wins'],
                    'ranked_games': stats_snapshot['ranked_games'],
                    'placements': stats_snapshot['placements'],
                    'league_elo': float(league.current_elo),
                    'total_rounds': stats_snapshot['total_rounds'],
                    'round_wins': stats_snapshot['round_wins'],
                    'round_placements': stats_snapshot['round_placements'],
                }
                if loss_dict.get('policy_entropy') is not None:
                    m['policy_entropy'] = float(loss_dict['policy_entropy'])
                broadcast('training_metrics', m)

                # --- Structured training log ---
                _log_learner.info(
                    "STEP %d | loss=%.4f (V:%.4f R:%.4f P:%.4f F:%.4f Rec:%.4f) | LR=%.2e | games=%d | rounds=%d | ELO=%.0f",
                    step, loss_dict['total'],
                    loss_dict.get('value', 0), loss_dict.get('reward', 0),
                    loss_dict.get('policy', 0), loss_dict.get('focus', 0),
                    loss_dict.get('recon', 0),
                    current_lr, game_receive_count,
                    stats_snapshot['total_rounds'], league.current_elo
                )

                # Verbose snapshot every 100 steps
                if step % 100 == 0:
                    wins = stats_snapshot['wins']
                    total_g = stats_snapshot['games']
                    wr = {k: (v / total_g * 100 if total_g > 0 else 0) for k, v in wins.items()}
                    _log_learner.info(
                        "SNAPSHOT step=%d | total_games=%d ranked=%d avg_len=%.1f | "
                        "win%%: P1=%.1f P2=%.1f P3=%.1f draw=%.1f | "
                        "placements: %s | rounds=%d round_wins=%s | "
                        "round_placements: %s | league_elo=%.1f | entropy=%s",
                        step, total_g, stats_snapshot['ranked_games'], stats_snapshot['avg_len'],
                        wr.get('1', 0), wr.get('2', 0), wr.get('3', 0), wr.get('draw', 0),
                        stats_snapshot['placements'],
                        stats_snapshot['total_rounds'], stats_snapshot['round_wins'],
                        stats_snapshot['round_placements'],
                        league.current_elo,
                        f"{loss_dict['policy_entropy']:.4f}" if loss_dict.get('policy_entropy') is not None else "N/A"
                    )


            # 2.5 Curriculum Logic
            # Update Loss only (no game count / win rate)
            if step % 10 == 0:
                curriculum.record_loss(loss_dict['total'])

            if step % 100 == 0:
                if curriculum.check_graduation(step=step, league=league):
                    new_stage = curriculum.advance(league=league)
                    if new_stage:
                        if shared_curriculum_stage is not None:
                            # Actors index stages by 0-based index; store current_stage_idx not stage_id (1-4)
                            shared_curriculum_stage.value = curriculum.current_stage_idx
                        broadcast('curriculum_graduation', {
                            'stage_id': new_stage.stage_id,
                            'stage': str(new_stage),
                        })
                        _log_learner.info("GRADUATING to Stage %s: %s", new_stage.stage_id, new_stage)
                         
                        # 1. Clear Buffer (Mandatory due to shape change)
                        buffer_cmd_sync({'type': 'clear'})
                        print("[Learner] Replay buffer cleared.", flush=True)
                        
                        # 2. Reset KOTH
                        frozen_models = {}
                        koth_step_counter = 0
                        frozen_path = os.path.join(config.checkpoint_dir, 'frozen_models.pt')
                        atomic_torch_save({'models': {}, 'active_pid': koth_active_pid}, frozen_path)
                        
                        # 3. Save "Graduation Checkpoint"
                        grad_path = os.path.join(config.checkpoint_dir, f'stage_{new_stage.stage_id-1}_graduated.pt')
                        atomic_torch_save(_strip_compiled_prefix(model.state_dict()), grad_path)
                        
                        _log_learner.info("Saved graduation snapshot: %s", grad_path)
                        
                        # Update last_stage for loop check (use 0-based index)
                        last_stage = curriculum.current_stage_idx


            # 3. Sync Weights (shared memory + file backup)
            if step % 100 == 0:
                cpu_state = _strip_compiled_prefix(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
                # Fast path: push to shared memory (minimal lock hold time)
                if shared_model is not None and weights_version is not None:
                    lock = weights_lock if weights_lock is not None else None
                    acquired = False
                    try:
                        if lock:
                            acquired = lock.acquire(timeout=0.5)
                        else:
                            acquired = True
                        if acquired:
                            shared_model.load_state_dict(cpu_state)
                            weights_version.value += 1
                    except Exception as e:
                        _log.error("[Learner] Shared memory sync failed: %s", e)
                    finally:
                        if lock and acquired:
                            lock.release()
                # Slow path: file backup (for resume, outside lock)
                atomic_torch_save(cpu_state, weights_path)
                if memory_bank:
                    atomic_torch_save(memory_bank.state_dict(), memory_path)

            # 4. Checkpoint (interval from config, fallback 500 to avoid save every step)
            ckpt_interval = getattr(config, 'checkpoint_interval', 500) or 500
            if step > 0 and step % ckpt_interval == 0:
                # Full checkpoint
                data = {
                    'step': step,
                    'model': _strip_compiled_prefix(model.state_dict()),
                    'network_state_dict': _strip_compiled_prefix(model.state_dict()), # For cross-compatibility with train.py
                    'optimizer': optimizer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),  # Cross-compat alias
                    'scaler': scaler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),  # Cross-compat alias
                    'global_step': step,  # Cross-compat alias
                    'stats': shared_stats.get_info(),
                    'curriculum_state': curriculum.state_dict(),
                    'league_current_elo': league.current_elo,
                    'pbt_active_agent_idx': active_agent_idx if population else 0,
                    'pbt_step_counter': pbt_step_counter if population else 0,
                    'koth_active_pid': koth_active_pid if config.koth_mode else 1,
                    'koth_step_counter': koth_step_counter if config.koth_mode else 0,
                    'frozen_models': frozen_models if config.koth_mode else {}
                }
                atomic_torch_save(data, latest_ckpt)
                _log_learner.info("CHECKPOINT step=%d saved to %s | games=%d rounds=%d elo=%.1f koth_pid=%s",
                                  step, latest_ckpt, stats_snapshot['games'] if 'stats_snapshot' in dir() else 0,
                                  stats_snapshot['total_rounds'] if 'stats_snapshot' in dir() else 0,
                                  league.current_elo,
                                  koth_active_pid if config.koth_mode else "N/A")
                
                # Save ReplayBuffer
                try:
                    buffer_path = os.path.join(config.checkpoint_dir, 'replay_buffer.pkl')
                    res = buffer_cmd_sync({'type': 'save', 'path': buffer_path}, timeout=600)
                    if res and res.get('type') == 'save_ok':
                        report = res.get('report', {})
                        _log_learner.info("BUFFER saved: %d games, %.2f/%.0f GB, avg_q=%.3f",
                                          report.get('num_games', 0), report.get('total_memory_gb', 0),
                                          report.get('max_memory_gb', 0), report.get('avg_quality', 0))
                except Exception as e:
                    _log_learner.error("Failed to save replay buffer: %s", e)

                # Save Metrics Log
                metrics_path = os.path.join(config.checkpoint_dir, 'metrics_log.jsonl')
                try:
                    save_metrics_log(metrics_path)
                    _log_learner.info("Saved metrics log (%d entries) to %s", len(METRICS_LOG), metrics_path)
                except Exception as e:
                    _log_learner.error("Failed to save metrics log: %s", e)

            # 5. League Snapshot (for self-play league)
            league_interval = getattr(config, 'league_save_interval', 5000)
            if step > 0 and step % league_interval == 0:
                league_dir = os.path.join(config.checkpoint_dir, 'league')
                os.makedirs(league_dir, exist_ok=True)
                league_path = os.path.join(league_dir, f'league_step_{step}.pt')
                try:
                    league_state = _strip_compiled_prefix(
                        {k: v.cpu() for k, v in model.state_dict().items()}
                    )
                    league_path = os.path.join(league_dir, f'league_step_{step}.pt')
                    atomic_torch_save(league_state, league_path)
                    _log_learner.info("Saved league snapshot: %s", league_path)
                    
                    # Add to League Manager
                    league.add_opponent(league_path, step)
                    
                    # Prune old snapshots (Managed by LeagueManager now? 
                    # LeagueManager manages the list. But deleting files is tricky.
                    # LeagueManager has max_snapshots. 
                    # Implementation in league.py just removes from list.
                    # We should let LeagueManager handle file deletion if it wants, or keep them.
                    # For now, keep files.

                    # Prune old snapshots
                    max_snaps = getattr(config, 'league_max_snapshots', 10)
                    snapshots = sorted(
                        [f for f in os.listdir(league_dir) if f.startswith('league_step_')],
                        key=lambda f: int(f.split('_')[-1].split('.')[0])
                    )
                    while len(snapshots) > max_snaps:
                        old = snapshots.pop(0)
                        os.remove(os.path.join(league_dir, old))
                        print(f"[Learner] Pruned old league snapshot: {old}")
                except Exception as e:
                    _log.error("[Learner] Failed to save league snapshot: %s", e)


        print("[Learner] Training finished.")

        # ── Graceful shutdown: save final state ──
        try:
            final_state = _strip_compiled_prefix(model.state_dict())
            final_data = {
                'step': step,
                'model': final_state,
                'network_state_dict': final_state, # For cross-compatibility
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'stats': shared_stats.get_info(),
                'curriculum_state': curriculum.state_dict(),
                'league_current_elo': league.current_elo,
                'pbt_active_agent_idx': active_agent_idx if population else 0,
                'pbt_step_counter': pbt_step_counter if population else 0,
                'koth_active_pid': koth_active_pid if config.koth_mode else 1,
                'koth_step_counter': koth_step_counter if config.koth_mode else 0,
                'frozen_models': frozen_models if config.koth_mode else {}
            }
            atomic_torch_save(final_data, latest_ckpt)
            buffer_path = os.path.join(config.checkpoint_dir, 'replay_buffer.pkl')
            buffer_cmd_sync({'type': 'save', 'path': buffer_path}, timeout=600)
            print(f"[Learner] Final checkpoint saved at step {step}.")
        except Exception as e:
            _log.warning("[Learner] Final checkpoint save failed: %s", e)

        # Cleanup buffer process and prefetcher
        try:
            buffer_cmd_sync({'type': 'stop'}, timeout=5)
            p_buffer.join(timeout=5)
            if p_buffer.is_alive():
                p_buffer.terminate()
        except Exception:
            pass

        try:
            p_reanalyze.join(timeout=5)
            if p_reanalyze.is_alive():
                p_reanalyze.terminate()
        except Exception:
            pass

        try:
            prefetcher.shutdown(wait=False)
        except Exception:
            pass

        stop_event.set()

    except Exception as e:
        _log.error("[Learner] Crash: %s", e)
        traceback.print_exc()
        # Attempt emergency checkpoint save
        try:
            emergency_state = _strip_compiled_prefix(model.state_dict())
            atomic_torch_save({
                'step': step, 'model': emergency_state,
                'stats': shared_stats.get_info()
            }, latest_ckpt)
            print(f"[Learner] Emergency checkpoint saved at step {step}.")
        except Exception:
            pass
        try:
            prefetcher.shutdown(wait=False)
        except Exception:
            pass
        stop_event.set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--actors', type=int, default=min(8, max(1, os.cpu_count() // 4)))
    parser.add_argument('--ws-port', type=int, default=5001)
    parser.add_argument('--resume', action='store_true')
    # Config overrides
    parser.add_argument('--min-buffer', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--simulations', type=int)
    parser.add_argument('--board-size', type=int)
    parser.add_argument('--max-memory', type=float, help='Max memory for replay buffer in GB (default: 35)')
    parser.add_argument('--auto-curriculum', action='store_true', help='Enable automatic curriculum progression')
    parser.add_argument('--final-stage-only', action='store_true', help='Skip curriculum: train only on final stage (100x100, 8-in-a-row); overrides --auto-curriculum')
    parser.add_argument('--koth-mode', action='store_true', help='Enable King-of-the-Hill rotational training')
    parser.add_argument('--koth-period', type=int, default=10000, help='Steps per KOTH rotation')
    parser.add_argument('--checkpoint-interval', type=int, help='Steps between checkpoints')
    
    # PBT Settings
    parser.add_argument('--pbt-pop-size', type=int, help='Population size for PBT')
    parser.add_argument('--pbt-period', type=int, help='Steps between PBT evolution events')
    parser.add_argument('--pbt-mutation-rate', type=float, help='Mutation rate for PBT hyperparameters')
    
    parser.add_argument('--ipc', action='store_true', help='Enable ZeroMQ IPC Batched Inference for Actors')
    
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)

    config = MuZeroConfig()
    if args.min_buffer: config.min_buffer_size = args.min_buffer
    if args.batch_size: config.batch_size_start = config.batch_size_end = args.batch_size
    if args.lr: config.learning_rate = args.lr
    if args.simulations: config.num_simulations_start = config.num_simulations_end = args.simulations
    if args.board_size: config.board_size = args.board_size
    if args.max_memory: config.max_memory_gb = args.max_memory
    config.auto_curriculum = args.auto_curriculum
    if getattr(args, 'final_stage_only', False):
        config.board_size = 100
        config.win_length = 8
        config.auto_curriculum = False
        print("[Config] --final-stage-only: 100x100, 8-in-a-row, no curriculum.")
    config.koth_mode = args.koth_mode
    if args.ipc:
        config.use_ipc_inference = True
        print("[Config] IPC Batched Inference Enabled via CLI.")
    if args.koth_period: config.koth_period = args.koth_period
    if args.checkpoint_interval: config.checkpoint_interval = args.checkpoint_interval
    
    # PBT Overrides
    if args.pbt_pop_size: config.pbt_population_size = args.pbt_pop_size
    if args.pbt_period: config.pbt_period = args.pbt_period
    if args.pbt_mutation_rate: config.pbt_mutation_rate = args.pbt_mutation_rate
    
    config.checkpoint_dir = 'checkpoints_async'
    config.validate()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    weights_path = os.path.join(config.checkpoint_dir, 'shared_weights.pt')
    memory_path = os.path.join(config.checkpoint_dir, 'shared_memory.pt')
    
    game_queue = mp.Queue(maxsize=config.game_queue_maxsize)
    stop_event = mp.Event()
    shared_stats = SharedStats()
    shared_curriculum_stage = mp.Value('i', 0)

    # Pre-load curriculum stage before spawning actors (so actors use correct board size from first game)
    if args.resume and getattr(config, 'auto_curriculum', False):
        latest_ckpt = os.path.join(config.checkpoint_dir, 'latest.pt')
        if os.path.exists(latest_ckpt):
            try:
                data = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
                if isinstance(data, dict) and 'curriculum_state' in data:
                    cs = data['curriculum_state']
                    stage_idx = int(cs.get('current_stage_idx', 0))
                    shared_curriculum_stage.value = stage_idx
                    print(f"[Main] Pre-loaded curriculum stage {stage_idx + 1} for actors (before spawn).")
                elif isinstance(data, dict) and 'curriculum_stage' in data:
                    raw = int(data['curriculum_stage'])
                    # Support both 0-based index (0-3) and legacy stage_id (1-4)
                    stage_idx = (raw - 1) if 1 <= raw <= 4 else raw
                    shared_curriculum_stage.value = min(3, max(0, stage_idx))
                    print(f"[Main] Pre-loaded curriculum_stage (index {shared_curriculum_stage.value}) for actors.")
            except Exception as e:
                print(f"[Main] Could not pre-load curriculum (actors may produce wrong-sized games briefly): {e}")

    live_queue = mp.Queue(maxsize=config.live_queue_maxsize)

    # ── Shared-memory model for zero-I/O weight sync (Opt 7) ──
    shared_model = MuZeroNetwork(config).to('cpu')
    shared_model.share_memory()
    # Version counter: actors check this to know when weights changed
    weights_version = mp.Value('i', 0)
    # Lock to prevent actors from reading partial state during learner writes
    weights_lock = mp.Lock()

    print(f"Starting Async Training with {args.actors} Actors...")

    # Start Actors
    processes = []
    for i in range(args.actors):
        lq = live_queue if i == 0 else None
        p = mp.Process(target=actor_loop,
                       args=(i, config, game_queue, weights_path, memory_path,
                             stop_event, shared_stats, lq, shared_model, weights_version, weights_lock, shared_curriculum_stage))
        p.start()
        processes.append(p)
    
    p_learner = mp.Process(target=learner_loop,
                           args=(config, args, game_queue, weights_path, memory_path,
                                 stop_event, shared_stats, live_queue, shared_model, weights_version, weights_lock, shared_curriculum_stage))
    p_learner.start()
    processes.append(p_learner)

    try:
        p_learner.join()
        # If learner finishes, stop actors
        stop_event.set()
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    main()
