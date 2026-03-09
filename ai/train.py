"""MuZero training loop with DeepSeek MLA, Gumbel MuZero, and Engram Memory.

See ai/MAINTENANCE.md for per-component maintainability notes.

Robustness and reliability checklist (do not remove safeguards when refactoring).
These safeguards support reliability: consistent, correct behavior under failure and bad input.
Safety: path validation for model/checkpoint I/O; bounded loops and config.validate(); input validation at API and train_step.
- server: load_model try/except; get_move validates board/player/cells and wraps inference in try/except with random-legal fallback.
- replay_buffer: _load_v2 validates entries; _ensure_precomputed/_filter_corrupted_games tolerate bad games; weight sum uses max(sum, 1e-9).
- train: resume in try/except; train_step validates batch keys and tensors for NaN/Inf, clamps targets, checks loss/grad NaN, uses config.max_grad_norm; warmup when warmup_steps > 0.
- train_async: full resume in try/except; NaN streak and re-sanitize; config.validate() at entry; queue.Full logged when repeated.
- self_play / game_env / mcts: step try/except and retry; action/center validation; value/reward sanitization.
- league / pbt / curriculum / board_render / data_augment: save retry; load with .get and skip bad entries; stage_idx clamp; entry validation.
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import os
import time
import json
import threading
import asyncio
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Optional

# Add parent dir to path so we can import ai package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.replay_buffer import ReplayBuffer
from ai.self_play import run_selfplay
from ai.data_augment import apply_board_augment
from ai.mcts import gumbel_muzero_search
from ai.engram import MemoryBank
from ai.league import LeagueManager
from ai.curriculum import CurriculumManager
from ai.log_utils import get_logger

_log = get_logger(__name__)

# ============================================================
#  WebSocket Dashboard Server
# ============================================================
WS_CLIENTS = set()
WS_LOOP = None

# Metrics History
METRICS_LOG = []

def save_metrics_log(path):
    """Persist METRICS_LOG to JSONL file (one JSON object per line)."""
    with open(path, 'w') as f:
        for m in METRICS_LOG:
            f.write(json.dumps(m) + '\n')

def load_metrics_log(path):
    """Load METRICS_LOG from JSONL file; returns the loaded list."""
    global METRICS_LOG
    METRICS_LOG.clear()
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    METRICS_LOG.append(json.loads(line))
                except: pass
    return METRICS_LOG

def start_ws_server(port: int = 5001):
    """Start WebSocket server in background thread. Prints only after server is bound or on error."""
    try:
        import websockets
        import websockets.server
    except ImportError:
        print("[Dashboard] websockets not installed, run: pip install websockets")
        return

    started_ok = threading.Event()

    async def handler(websocket):
        WS_CLIENTS.add(websocket)
        try:
            # Send history on connect
            if METRICS_LOG:
                await websocket.send(json.dumps({'type': 'metrics_history', 'data': METRICS_LOG}))
            
            async for _ in websocket: pass
        except: pass
        finally: WS_CLIENTS.discard(websocket)

    async def run_server():
        global WS_LOOP
        WS_LOOP = asyncio.get_event_loop()
        try:
            async with websockets.serve(handler, "0.0.0.0", port):
                started_ok.set()
                await asyncio.Future()
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98 or e.errno == 10048:
                print(f"[Dashboard] 端口 {port} 已被占用，仪表盘不可用。可指定其它端口: python -m ai.train_async --ws-port 5002")
            else:
                print(f"[Dashboard] WebSocket 启动失败: {e}")

    def _thread_target():
        asyncio.run(run_server())

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    if started_ok.wait(timeout=3.0):
        print(f"[Dashboard] WebSocket 已就绪 ws://0.0.0.0:{port} — 打开 train_dashboard.html 并连接此地址即可监控")
    # else: server may still be starting or port in use (run_server will print)


def broadcast(event_type: str, data: dict):
    if not WS_CLIENTS or WS_LOOP is None: return
    msg = json.dumps({'type': event_type, **data})
    
    # Also log metrics if relevant
    if event_type == 'training_metrics':
        METRICS_LOG.append(data)

    async def _send():
        for ws in WS_CLIENTS.copy():
            try: await ws.send(msg)
            except: pass
    try: asyncio.run_coroutine_threadsafe(_send(), WS_LOOP)
    except: pass


# ============================================================
#  Training
# ============================================================

def train(config: MuZeroConfig, args):
    """Main training loop."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("  MuZero Training (DeepSeek MLA + Gumbel + Engram)")
    print("=" * 60)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        # Enable cuDNN autotuner for fixed-size conv inputs
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    
    start_ws_server(port=args.ws_port)

    # Create network
    network = MuZeroNetwork(config).to(device)
    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {param_count:,}")

    # Create opponent network (inactive, used for League/Eval)
    opponent_network = MuZeroNetwork(config).to(device)
    opponent_network.eval()

    # Optimizer
    # Fused AdamW: single CUDA kernel per step
    _fused = device.type == 'cuda'
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=_fused
    )
    # Sync train: StepLR (+ optional warmup). Async train uses LambdaLR (warmup + cosine).
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_decay_steps, gamma=config.lr_decay_rate
    )

    # Replay buffer & Memory Bank
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        max_memory_gb=config.max_memory_gb,
        priority_alpha=config.priority_alpha,
        min_games=config.min_buffer_games
    )
    memory_bank = None
    if config.use_engram:
        memory_bank = MemoryBank(capacity=config.memory_capacity, 
                                 key_dim=config.hidden_state_dim,
                                 value_dim=config.hidden_state_dim)
        print(f"Engram Memory initialized (capacity: {config.memory_capacity})")

    # Curriculum & League
    curriculum = CurriculumManager(config)
    league = LeagueManager(config)
    
    # Apply initial curriculum stage
    stage = curriculum.get_current_stage()
    config.board_size = stage.board_size
    config.win_length = stage.win_length
    print(f"Curriculum Stage {stage.stage_id}: {stage.board_size}x{stage.board_size}, Win {stage.win_length}")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    global_step = 0
    selfplay_iteration = 0
    best_loss = float('inf')
    total_games = 0
    win_counts = {1: 0, 2: 0, 3: 0, 'draw': 0}

    # Resume
    if args.resume:
        ckpt_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
        # Also try train_async.py's checkpoint name
        if not os.path.exists(ckpt_path):
            alt_path = os.path.join(config.checkpoint_dir, "latest.pt")
            if os.path.exists(alt_path):
                ckpt_path = alt_path
        if os.path.exists(ckpt_path):
            try:
                print(f"Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                # Cross-compatible model key: try 'network_state_dict' then 'model'
                state_dict = checkpoint.get('network_state_dict', checkpoint.get('model'))
                if state_dict is None:
                    raise KeyError("Checkpoint missing 'network_state_dict' or 'model' key")

                model_dict = network.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items()
                                  if k in model_dict and v.shape == model_dict[k].shape}

                if len(pretrained_dict) < len(state_dict):
                    _log.warning(
                        "%d keys skipped due to shape mismatch (likely value head or aux heads).",
                        len(state_dict) - len(pretrained_dict)
                    )
                model_dict.update(pretrained_dict)
                network.load_state_dict(model_dict)

                # Cross-compatible optimizer key: try 'optimizer_state_dict' then 'optimizer'
                opt_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer'))
                if opt_state:
                    try:
                        optimizer.load_state_dict(opt_state)
                    except (ValueError, RuntimeError) as e:
                        _log.warning("Optimizer state incompatible (architecture changed), resetting: %s", e)

                # Cross-compatible scaler key: try 'scaler_state_dict' then 'scaler'
                scaler_state = checkpoint.get('scaler_state_dict', checkpoint.get('scaler'))
                if scaler_state:
                    try:
                        scaler.load_state_dict(scaler_state)
                    except Exception as e:
                        _log.warning("Scaler state incompatible, resetting: %s", e)

                # Cross-compatible step key: try 'global_step' then 'step'
                global_step = checkpoint.get('global_step', checkpoint.get('step', 0))
                selfplay_iteration = checkpoint.get('selfplay_iteration', 0)
                best_loss = checkpoint.get('best_loss', float('inf'))

                # Cross-compatible stats: try flat keys then 'stats' dict
                total_games = checkpoint.get('total_games', 0)
                if total_games == 0 and 'stats' in checkpoint:
                    total_games = checkpoint['stats'].get('games', 0)

                if memory_bank and 'memory_bank' in checkpoint:
                    memory_bank.load_state_dict(checkpoint['memory_bank'])
                    print(f"  Memory bank loaded: {memory_bank.count} entries")

                if 'curriculum_state' in checkpoint:
                    curriculum.load_state_dict(checkpoint['curriculum_state'])
                elif 'curriculum_stage' in checkpoint:
                    stage_idx = checkpoint['curriculum_stage']
                    curriculum.set_stage(stage_idx)
                else:
                    _log.warning("No curriculum_state or curriculum_stage in checkpoint; using default stage.")

                if 'league_current_elo' in checkpoint:
                    league.current_elo = checkpoint['league_current_elo']
                    print(f"  League Elo restored: {league.current_elo:.1f}")
                else:
                    _log.warning("league_current_elo missing in checkpoint; keeping default 1200.")

                buf_path = os.path.join(config.checkpoint_dir, "replay_buffer.pkl")
                if os.path.exists(buf_path):
                    replay_buffer.load(buf_path)

                print(f"Resumed at step={global_step}, games={total_games}")
            except Exception as e:
                _log.error("Resume failed: %s: %s. Use --no-resume or check checkpoint file.", type(e).__name__, e)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    target_steps = global_step + args.steps
    t_start = time.time()

    while global_step < target_steps:
        iter_start = time.time()
        selfplay_iteration += 1

        # ---- Phase 1: Self-play ----
        print(f"Iteration {selfplay_iteration} | Step {global_step}/{target_steps}")
        print(f"   Self-play: generating {config.selfplay_games_per_iter} games...", end='', flush=True)

        sp_start = time.time()
        network.eval()
        
        current_min = config.min_buffer_size if global_step == 0 else config.min_buffer_games
        if replay_buffer.num_games() < current_min:
            # Warmup phase: strictly self-play to fill buffer
            nets = network
            is_eval = False
        else:
            # League/Eval Probability
            if np.random.rand() < getattr(config, 'league_opponent_prob', 0.2):
                opp = league.get_opponent()
                if opp and os.path.exists(opp.checkpoint_path):
                    # Load opponent
                    try:
                        ckpt = torch.load(opp.checkpoint_path, map_location=device, weights_only=False)
                        # Filter matching keys only (in case architecture changed slightly)
                        opp_state = ckpt['network_state_dict']
                        curr_state = opponent_network.state_dict()
                        common_state = {k: v for k, v in opp_state.items() if k in curr_state and v.shape == curr_state[k].shape}
                        opponent_network.load_state_dict(common_state, strict=False)
                        
                        # Setup Match: PID 1 = Current, Others = Opponent
                        # 3-player game: 1 vs 2 & 3
                        nets = {1: network}
                        for p in range(2, config.num_players + 1):
                            nets[p] = opponent_network
                        is_eval = True
                        print(f" [League] Playing against opponent from step {opp.step} (Elo {opp.elo:.0f})")
                    except Exception as e:
                        _log.warning("League: Failed to load opponent %s: %s", opp.checkpoint_path, e)
                        nets = network
                        is_eval = False
                else:
                    nets = network
                    is_eval = False
            else:
                nets = network
                is_eval = False

        # We pass memory_bank to run_selfplay if we want memory usage during play
        # (Assuming self_play.py updated to handle it, or we skip for now)
        games = run_selfplay(
            nets, config, config.selfplay_games_per_iter,
            broadcast_fn=broadcast,
            iteration=selfplay_iteration,
        )
        
        # Curriculum: record one game result per game (PID 1 = current agent)
        for g in games:
            score = 1.0 if g.winner == 1 else (0.5 if g.winner is None else 0.0)
            curriculum.record_game_result(score)

        # League: update Elo when this was a league eval
        if is_eval:
            wins = sum(1 for g in games if g.winner == 1)
            draws = sum(1 for g in games if g.winner is None)
            count = len(games)
            wr = (wins + 0.5 * draws) / max(1, count)
            league.record_match(opp, wr)
            print(f"   [Eval] Result vs Elo {opp.elo:.0f}: WR {wr*100:.1f}% -> New Elo {league.current_elo:.0f}")

        for game in games:
            replay_buffer.save_game(game)
            if memory_bank:
                _update_memory_bank(memory_bank, game, network, device, config)

        sp_time = time.time() - sp_start
        
        # Stats
        for g in games:
            total_games += 1
            w = g.winner
            if w and w in win_counts: win_counts[w] += 1
            else: win_counts['draw'] += 1

        print(f" done ({sp_time:.1f}s)")

        # ---- Phase 2: Training ----
        current_min = config.min_buffer_size if global_step == 0 else config.min_buffer_games
        if replay_buffer.num_games() < current_min:
            continue

        network.train()
        losses = {'total': [], 'value': [], 'reward': [], 'policy': [], 'consistency': [], 'focus': [], 'recon': [], 'policy_entropy': []}

        steps_this_iter = min(config.training_steps_per_iter, target_steps - global_step)
        print(f"   Training: {steps_this_iter} steps...", end='', flush=True)

        # Efficiency: sync train does not prefetch; use train_async for higher throughput.
        for _ in range(steps_this_iter):
            progress = min(1.0, global_step / max(1, getattr(config, 'progression_steps', 100000)))
            current_batch_size = int(getattr(config, 'batch_size_start', 128) + (getattr(config, 'batch_size_end', 1024) - getattr(config, 'batch_size_start', 128)) * progress)
            
            batch = replay_buffer.sample_batch(
                batch_size=current_batch_size,
                num_unroll_steps=config.num_unroll_steps,
                td_steps=config.td_steps,
                discount=config.discount,
                action_size=config.policy_size,
                view_size=config.local_view_size
            )
            if getattr(config, 'augment_board', False):
                apply_board_augment(batch, np.random.default_rng(), noise_std=getattr(config, 'augment_noise_std', 0.0) or None)
            l_dict = train_step(network, optimizer, scaler, batch, config, device, memory_bank,
                                step_counter=global_step)
            warmup_steps = getattr(config, 'warmup_steps', 0)
            if warmup_steps > 0 and global_step < warmup_steps:
                # Linear warmup: 0.01 * base_lr -> base_lr over warmup_steps
                lr = config.learning_rate * (0.01 + 0.99 * global_step / max(1, warmup_steps))
                optimizer.param_groups[0]['lr'] = lr
            else:
                scheduler.step()
            global_step += 1
            
            for k, v in l_dict.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v)

            if global_step % config.checkpoint_interval == 0:
                save_checkpoint(network, optimizer, scheduler, scaler, replay_buffer, memory_bank,
                                global_step, selfplay_iteration, best_loss, config, total_games, win_counts,
                                curriculum, league)

        avg_loss = np.mean(losses['total'])
        print(f" done. Loss: {avg_loss:.4f} (C={np.mean(losses['consistency']):.4f}, F={np.mean(losses.get('focus', [0])): .4f}, Rec={np.mean(losses.get('recon', [0])): .4f})")
        
        curriculum.record_loss(avg_loss)
        
        if curriculum.check_graduation(step=global_step, league=league):
            stage = curriculum.advance(league=league)
            if stage:
                broadcast('curriculum_graduation', {'stage_id': stage.stage_id, 'stage': str(stage)})
                config.board_size = stage.board_size
                config.win_length = stage.win_length
                print(f"*** ADVANCED TO STAGE {stage.stage_id} ***")
                # Clear buffer for new stage?
                replay_buffer.clear()
        
        # League Update
        if global_step % config.league_save_interval == 0:
            league.add_opponent(os.path.join(config.checkpoint_dir, "latest_checkpoint.pt"), global_step)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(network.state_dict(), os.path.join(config.checkpoint_dir, "best_model.pt"))
            
        # Broadcast metrics
        metrics = {
            'step': global_step,
            'loss': float(avg_loss),
            'loss_value': float(np.mean(losses['value'])),
            'loss_reward': float(np.mean(losses['reward'])),
            'loss_policy': float(np.mean(losses['policy'])),
            'loss_consistency': float(np.mean(losses['consistency'])),
            'loss_focus': float(np.mean(losses.get('focus', [0]))),
            'loss_recon': float(np.mean(losses.get('recon', [0]))),
            'lr': scheduler.get_last_lr()[0],
            'total_games': total_games,
            'win_counts': {str(k): v for k, v in win_counts.items()},
        }
        if losses.get('policy_entropy'):
            metrics['policy_entropy'] = float(np.mean(losses['policy_entropy']))
        broadcast('training_metrics', metrics)

    # Final save
    # Final save
    save_checkpoint(network, optimizer, scheduler, scaler, replay_buffer, memory_bank,
                    global_step, selfplay_iteration, best_loss, config, total_games, win_counts,
                    curriculum, league)


def _update_memory_bank(memory_bank, game, network, device, config):
    """Save key positions from game to memory bank. Processes in fixed-size chunks to avoid GPU OOM and recompile thrashing."""
    # Store e.g. every 5th step
    indices = list(range(0, len(game), 5))
    if len(indices) == 0:
        return

    from ai.game_env import EightInARowEnv
    from ai.self_play import _get_observation_and_mask
    import numpy as np

    env = EightInARowEnv(board_size=getattr(game, 'board_size', config.board_size),
                         win_length=getattr(game, 'win_length', config.win_length))
    env.reset()
    h_half = config.local_view_size // 2
    obs_dict = {}

    max_idx = max(indices)
    for i in range(max_idx + 1):
        if i in indices:
            ctr = game.centers[i]
            rotated = env._get_rotated_planes_cached()
            _, legal_mask = env.get_legal_moves_and_mask(ctr[0], ctr[1], config.local_view_size)
            obs, _, _ = _get_observation_and_mask(env, config, ctr[0], ctr[1], legal_mask, rotated)
            obs_dict[i] = obs

        act = game.actions[i]
        ctr = game.centers[i]
        lr, lc = act // config.local_view_size, act % config.local_view_size
        env.step(ctr[0] - h_half + lr, ctr[1] - h_half + lc)

    # Chunk size to avoid VRAM spike and dynamic-shape recompiles (fixed batch size per chunk)
    chunk_size = getattr(config, 'memory_bank_chunk_size', 64)
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            chunk_indices = indices[start : start + chunk_size]
            obs_batch = np.array([obs_dict[i] for i in chunk_indices])
            obs_tensor = torch.from_numpy(obs_batch).to(device)

            hidden_states = network.representation(obs_tensor)

            values = [game.root_values[i] for i in chunk_indices]
            rewards = [game.rewards[i] for i in chunk_indices]
            val_batch = np.array(values, dtype=np.float32)
            rew_batch = np.array(rewards, dtype=np.float32)
            val_tensor = torch.from_numpy(val_batch).to(device)
            rew_tensor = torch.from_numpy(rew_batch).to(device)

            vals = torch.zeros(len(chunk_indices), config.hidden_state_dim, device=device)
            vals[:, 0] = val_tensor[:, 0]
            vals[:, 1] = rew_tensor

            memory_bank.write(hidden_states, vals)


def train_step(network, optimizer, scaler, batch, config, device, memory_bank,
               step_counter: int = 0, accumulate: bool = False,
               accum_scale: float = 1.0, koth_active_pid: Optional[int] = None):
    """Single training step with Consistency Loss and Engram.

    Args:
        accumulate: if True, only backward (accumulate gradients) without
                    optimizer.step(). Caller is responsible for stepping.
    """
    # Robustness: skip step if batch is missing required keys (e.g. bad producer)
    required_keys = (
        'observations', 'actions', 'target_values', 'target_rewards', 'target_policies',
        'global_states', 'target_centers', 'target_threats', 'target_opponent_actions', 'target_heatmaps',
    )
    if not isinstance(batch, dict) or any(k not in batch for k in required_keys):
        missing = [] if isinstance(batch, dict) else ['batch not a dict']
        if isinstance(batch, dict):
            missing = [k for k in required_keys if k not in batch]
        _log.warning("Skipping step: batch missing required key(s): %s", missing)
        return {'total': float('nan'), 'value': 0.0, 'reward': 0.0,
                'policy': 0.0, 'consistency': 0.0, 'focus': 0.0, 'policy_entropy': 0.0, '_nan': True}

    # Support both pre-converted tensors (pinned, from prefetcher) and raw numpy
    # Efficiency: non_blocking for pinned→GPU async DMA when batch is prefetched.
    _nb = device.type == 'cuda'
    def _to_dev(key, dtype=None):
        v = batch[key]
        if isinstance(v, torch.Tensor):
            t = v.to(device, non_blocking=_nb)
        else:
            t = torch.from_numpy(v).to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    obs = _to_dev('observations')
    actions = _to_dev('actions')
    target_values = _to_dev('target_values')
    target_rewards = _to_dev('target_rewards')
    target_policies = _to_dev('target_policies')
    
    # Focus Network Inputs
    # Focus Network Inputs
    global_states = _to_dev('global_states')
    target_centers = _to_dev('target_centers', dtype=torch.long)

    # Aux Targets (Phase 2)
    target_threats = _to_dev('target_threats')
    target_opponent_actions = _to_dev('target_opponent_actions', dtype=torch.long)
    target_heatmaps = _to_dev('target_heatmaps')

    # Session Context (Phase 4)
    session_contexts = None
    if 'session_contexts' in batch:
        session_contexts = _to_dev('session_contexts')

    # ── Batch data NaN/Inf validation (every 50 steps to reduce overhead) ──
    # The per-step loss NaN guard below catches any NaN that slips through.
    if step_counter % 50 == 0:
        validation_tensors = [('obs', obs), ('target_values', target_values),
                              ('target_rewards', target_rewards), ('target_policies', target_policies),
                              ('global_states', global_states), ('target_heatmaps', target_heatmaps),
                              ('target_threats', target_threats), ('target_opponent_actions', target_opponent_actions)]
        if session_contexts is not None:
            validation_tensors.append(('session_contexts', session_contexts))
        for name, t in validation_tensors:
            if torch.isnan(t).any() or torch.isinf(t).any():
                return {'total': float('nan'), 'value': 0.0, 'reward': 0.0,
                        'policy': 0.0, 'consistency': 0.0, 'focus': 0.0, '_nan': True}

    # Phase 6: KOTH Gradient Masking
    loss_mask = 1.0
    if config.koth_mode and 'player_ids' in batch:
        # Move player_ids to device
        player_ids = _to_dev('player_ids').long()  # (B, )
        # Mask: 1 if active, 0 if frozen. If koth_active_pid not provided, assume all active (or handle in learner)
        # Actually, caller should filter or we pass active_pid.
        # Let's add active_pid to arguments or config? 
        # Config is static. Argument is better.
        pass # Will implement via argument update below

    # Clamp target values/rewards to prevent extreme MSE loss under fp16
    target_values = target_values.clamp(-5.0, 5.0)
    target_rewards = target_rewards.clamp(-5.0, 5.0)

    if not accumulate:
        optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # Enable BF16 Mixed Precision for 30-50% speedup
        # 1. Initial Representation
        hidden_state = network.representation(obs)

        # Apply session context (Phase 4)
        hidden_state = network.apply_session_context(hidden_state, session_contexts)

        # Memory Retrieval (Root)
        mem_keys, mem_vals = None, None
        if memory_bank and memory_bank.count > 0:
            mem_keys, mem_vals = memory_bank.read(hidden_state, top_k=config.memory_top_k)
            hidden_state_aug = network.engram_module(hidden_state, mem_keys, mem_vals)
        else:
            hidden_state_aug = hidden_state
            
        # Prediction + Aux Heads
        policy_logits, value, threat_logits, opp_action_logits, heatmap_logits = network.prediction(hidden_state_aug)
        
        # --- Loss Calculation with Masking ---
        # If koth_active_pid is set, mask out samples from other players
        mask = torch.ones_like(target_values[:, 0, 0]) # (B,) assuming target_values is (B, K+1, 3)
        mask_bc = mask.unsqueeze(1) # (B, 1) for vector value
        
        if koth_active_pid is not None and 'player_ids' in batch:
             pids = _to_dev('player_ids').long()
             mask = (pids == koth_active_pid).float() # (B,)
             mask_bc = mask.unsqueeze(1)      # (B, 1)
        
        # Stability: avoid division by zero when batch is fully masked (e.g. KOTH)
        mask_sum = mask.sum().clamp(min=1.0)
        
        # Value Loss (Vector): MSE to n-step TD target (vector).
        loss_v_raw = F.mse_loss(value, target_values[:, 0], reduction='none') * mask_bc
        loss_v = loss_v_raw.sum() / mask_sum
        
        # Policy Loss: Cross-entropy to MCTS improved policy.
        log_probs = F.log_softmax(policy_logits, dim=1)
        # target_policies: (B, A). log_probs: (B, A). sum(dim=1) -> (B,)
        loss_p_raw = -(target_policies[:, 0] * log_probs).sum(dim=1) * mask
        loss_p = loss_p_raw.sum() / mask_sum

        # Track per-sample TD Error for Prioritized Experience Replay
        td_errors = (loss_v_raw.sum(dim=-1) + loss_p_raw).detach()

        # Interpretability: batch mean policy entropy (root step)
        probs = F.softmax(policy_logits, dim=1)
        policy_entropy = -(probs * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean().item()

        loss_r = torch.zeros_like(loss_v)
        loss_c = torch.zeros_like(loss_v)
        
        # --- State Reconstruction Loss ---
        recon_obs = network.reconstruct_state(hidden_state)
        # obs is (B, 8, 21, 21) where planes are binary floats
        loss_recon_raw = F.mse_loss(recon_obs, obs, reduction='none').mean(dim=[1, 2, 3]) * mask
        loss_recon = loss_recon_raw.sum() / mask_sum
        
        # --- Auxiliary Losses (calculated only at root step k=0) ---
        # 1. Threat Detection (BCE)
        loss_threat_raw = F.binary_cross_entropy_with_logits(threat_logits, target_threats, reduction='none').mean(dim=1) * mask
        loss_threat = loss_threat_raw.sum() / mask_sum
        
        # 2. Opponent Action (CrossEntropy with ignore_index=-1)
        loss_opp_raw = F.cross_entropy(opp_action_logits, target_opponent_actions, ignore_index=-1, reduction='none') * mask
        loss_opp = loss_opp_raw.sum() / mask_sum
        
        # 3. heatmap (BCE)
        # heatmap_logits: (B, 1, 21, 21), target: (B, 21, 21) → unsqueeze target
        loss_heatmap_raw = F.binary_cross_entropy_with_logits(heatmap_logits, target_heatmaps.unsqueeze(1), reduction='none').mean(dim=[1,2,3]) * mask
        loss_heatmap = loss_heatmap_raw.sum() / mask_sum
        
        # Scales for aux losses (adjust as needed)
        w_threat = 1.0
        w_opp = 0.5 
        w_heatmap = 1.0
        
        loss_aux = w_threat * loss_threat + w_opp * loss_opp + w_heatmap * loss_heatmap
        
        # --- Focus Loss (Regression) ---
        # Network outputs (B, 2) in [0, 1] via Sigmoid
        focus_pred = network.focus_net(global_states)
        
        # Target centers are indices (0..9999)
        # Convert to (r, c) normalized
        w = global_states.shape[-1]
        target_r = (target_centers // w).float() / w
        target_c = (target_centers % w).float() / w
        target_coords = torch.stack([target_r, target_c], dim=1)
        
        # MSE Loss - Focus net runs on global state which is always valid, but maybe we mask it too?
        # Yes, focus net should also only train on active player's data.
        loss_f_raw = F.mse_loss(focus_pred, target_coords, reduction='none').mean(dim=1) * mask * 10.0
        loss_f = loss_f_raw.sum() / mask_sum
        
        # MuZero: scale gradient by 1/K for unroll.
        gradient_scale = 1.0 / config.num_unroll_steps
        curr_state = hidden_state_aug

        # Unroll
        for k in range(config.num_unroll_steps):
            # 2. Dynamics
            action_onehot = F.one_hot(actions[:, k].long(), config.policy_size).float()
            next_state, reward = network.dynamics(curr_state, action_onehot)
            
            # Scale gradient by 1/K (MuZero paper: reduce gradient magnitude through unroll)
            _gs = gradient_scale  # Capture in closure to avoid lambda variable issue
            next_state.register_hook(lambda grad, s=_gs: grad * s)

            # 3. Consistency Loss (EfficientZero)
            # Apply on K=0 to anchor the dynamics trajectory
            if config.use_consistency and k == 0:
                next_obs = _to_dev('next_observations')
                with torch.no_grad():
                    h_real = network.representation(next_obs)
                # Compare projected predicted state vs projected real state
                loss_c_raw = network.consistency(next_state, h_real, reduction='none') * gradient_scale * mask
                loss_c += loss_c_raw.sum() / mask_sum

            # --- Lightweight Value Reanalyze ---
            # Blend stale historical MCTS target with fresh network prediction
            # This provides Reanalyze benefits (updating stale targets) with zero MCTS cost
            with torch.no_grad():
                fresh_value = value.detach()
                reanalyzed_target = 0.5 * target_values[:, k + 1] + 0.5 * fresh_value
                
            loss_v_raw = F.mse_loss(value, reanalyzed_target, reduction='none') * mask_bc
            loss_v += (loss_v_raw.sum() / mask_sum) * gradient_scale
            
            loss_r_raw = F.mse_loss(reward, target_rewards[:, k], reduction='none') * mask
            loss_r += (loss_r_raw.sum() / mask_sum) * gradient_scale
            
            # Use unpacked values, ignoring aux at unrolled steps
            policy_logits, value, _, _, _ = network.prediction(next_state)
            
            # Inline policy loss with masking
            log_probs = F.log_softmax(policy_logits.float(), dim=-1).clamp(min=-30.0)
            loss_p_raw = -(target_policies[:, k + 1] * log_probs).sum(dim=-1) * mask
            loss_p += (loss_p_raw.sum() / mask_sum) * gradient_scale
            
            td_errors += (loss_v_raw.sum(dim=-1) + loss_r_raw + loss_p_raw).detach() * gradient_scale
            
            curr_state = next_state

        # Weighting: Focus loss should be significant but not overwhelm
        # Maybe 0.1 or 1.0 depending on magnitude. CE is typically 2-5 initially.
        # Let's use 0.5 for now.
        # Recon loss is MSE over binary planes, so it's bounded [0, 1]. We weight it to match other losses.
        total_loss = loss_v + loss_r + loss_p + config.lambda_consistency * loss_c + 0.5 * loss_f + loss_aux + 2.0 * loss_recon

    # ── NaN guard: skip step if loss is NaN/Inf ──
    # Always zero_grad on NaN regardless of accumulate, to prevent poison gradients
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        optimizer.zero_grad(set_to_none=True)
        return {'total': float('nan'), 'value': 0.0, 'reward': 0.0,
                'policy': 0.0, 'consistency': 0.0, 'focus': 0.0, 'policy_entropy': 0.0, 'recon': 0.0, '_nan': True}

    scaler.scale(total_loss * accum_scale).backward()

    if not accumulate:
        scaler.unscale_(optimizer)

        # ── Gradient NaN guard ──
        has_nan_grad = False
        for p in network.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break
        if has_nan_grad:
            optimizer.zero_grad(set_to_none=True)  # always clear poison gradients
            scaler.update()
            return {'total': float('nan'), 'value': 0.0, 'reward': 0.0,
                    'policy': 0.0, 'consistency': 0.0, 'focus': 0.0, 'policy_entropy': 0.0, 'recon': 0.0, '_nan': True, 'td_errors': None}

        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=getattr(config, 'max_grad_norm', 10.0))
        scaler.step(optimizer)
        scaler.update()

    return {'total': total_loss.item(), 'value': loss_v.item(),
            'reward': loss_r.item(), 'policy': loss_p.item(),
            'consistency': loss_c.item(), 'focus': loss_f.item(),
            'recon': loss_recon.item(),
            'aux': loss_aux.item(), 'policy_entropy': policy_entropy,
            'td_errors': td_errors.cpu().numpy()}


def _policy_loss(logits, targets):
    # Compute in float32 to avoid fp16 overflow → -inf → NaN
    logits_f32 = logits.float()
    targets_f32 = targets.float()
    log_probs = F.log_softmax(logits_f32, dim=-1)
    # Clamp to prevent -inf * 0 = NaN (IEEE 754 pitfall)
    log_probs = log_probs.clamp(min=-30.0)
    return -torch.sum(targets_f32 * log_probs, dim=-1).mean()


def save_checkpoint(network, optimizer, scheduler, scaler, replay_buffer, memory_bank, 
                    step, iteration, best_loss, config, total_games, win_counts,
                    curriculum=None, league=None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt = {
        # Primary keys (train.py native)
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'global_step': step,
        'selfplay_iteration': iteration,
        'best_loss': best_loss,
        'config': vars(config),
        'total_games': total_games,
        'win_counts': win_counts,
        'memory_bank': memory_bank.state_dict() if memory_bank else None,
        # Cross-compatible aliases (for train_async.py resume)
        'model': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'stats': {
            'games': total_games,
            'wins': {
                '1': win_counts.get(1, 0),
                '2': win_counts.get(2, 0),
                '3': win_counts.get(3, 0),
                'draw': win_counts.get('draw', 0),
            },
        },
    }
    
    if curriculum:
        ckpt['curriculum_state'] = curriculum.state_dict()
    if league:
        ckpt['league_current_elo'] = league.current_elo
        
    torch.save(ckpt, os.path.join(config.checkpoint_dir, "latest_checkpoint.pt"))
    if step % 2000 == 0:
        torch.save(ckpt, os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"))
    
    # Async replay buffer save — runs in background to avoid blocking training
    path = os.path.join(config.checkpoint_dir, "replay_buffer.pkl")
    _async_save_replay_buffer(replay_buffer, path)


_replay_save_lock = threading.Lock()

def _async_save_replay_buffer(replay_buffer, path: str):
    """Save replay buffer in a background daemon thread, guarded by a lock."""
    if not _replay_save_lock.acquire(blocking=False):
        _log.info("Replay buffer save already in progress, skipping.")
        return
    
    def _save():
        try:
            replay_buffer.save(path)
        except Exception as e:
            _log.error("Async replay buffer save failed: %s", e)
        finally:
            _replay_save_lock.release()
    
    t = threading.Thread(target=_save, daemon=True)
    t.start()


# Main boilerplate...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--ws-port', type=int, default=5001)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--cpu', action='store_true')
    # Overrides
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--selfplay-games', type=int)
    parser.add_argument('--simulations', type=int)
    parser.add_argument('--min-buffer', type=int)
    # Curriculum
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], help="Curriculum stage (1=15x15, 2=30x30, 3=50x50, 4=100x100)")
    parser.add_argument('--board-size', type=int, help="Override board size")
    parser.add_argument('--win-length', type=int, help="Override win length")

    args = parser.parse_args()
    
    config = MuZeroConfig()
    
    # Curriculum Logic: Apply stage defaults first
    if args.stage:
        print(f"Applying Curriculum Stage {args.stage} defaults...")
        if args.stage == 1:
            config.board_size = 15
            config.win_length = 5
        elif args.stage == 2:
            config.board_size = 30
            config.win_length = 6
        elif args.stage == 3:
            config.board_size = 50
            config.win_length = 7
        elif args.stage == 4:
            config.board_size = 100
            config.win_length = 8
            
    # Apply overrides (if any)
    if args.board_size: config.board_size = args.board_size
    if args.win_length: config.win_length = args.win_length
    
    print(f"Configuration: Board {config.board_size}x{config.board_size}, Win Length {config.win_length}")

    if args.cpu: config.device = 'cpu'
    if args.batch_size: config.batch_size_start = config.batch_size_end = args.batch_size
    if args.lr: config.learning_rate = args.lr
    if args.selfplay_games: config.selfplay_games_per_iter = args.selfplay_games
    if args.simulations: config.num_simulations_start = config.num_simulations_end = args.simulations
    if args.min_buffer: config.min_buffer_size = args.min_buffer
    
    config.checkpoint_dir = args.checkpoint_dir

    if os.environ.get("STRICT_CONFIG", "").strip().lower() in ("1", "true", "yes"):
        config.validate()

    train(config, args)

if __name__ == '__main__':
    main()
