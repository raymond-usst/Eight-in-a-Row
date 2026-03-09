"""FastAPI server for MuZero AI player.

Maintainability: see ai/MAINTENANCE.md."""

import argparse
import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.muzero_config import MuZeroConfig
from ai.muzero_network import MuZeroNetwork
from ai.game_env import EightInARowEnv
from ai.mcts import gumbel_muzero_search, select_action
from ai.log_utils import get_logger
from ai.path_utils import safe_under

_log = get_logger(__name__)

# ---- App ----
app = FastAPI(title="Eight-in-a-Row MuZero AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
network: Optional[MuZeroNetwork] = None
config: Optional[MuZeroConfig] = None
device: Optional[torch.device] = None


class MoveRequest(BaseModel):
    """Request body for AI move."""
    board: List[List[int]]          # 100x100 board, 0=empty 1=red 2=green 3=blue
    current_player: int             # 1, 2, or 3
    move_history: List[List[int]]   # [[row, col, player], ...]


class MoveResponse(BaseModel):
    """Response with AI's chosen move."""
    row: int
    col: int
    confidence: float
    thinking_time_ms: float
    root_value: Optional[List[float]] = None  # [V_me, V_next, V_prev] from MCTS root
    top_actions: Optional[List[dict]] = None  # top 3 moves: [{"row", "col", "prob"}, ...]


class StatusResponse(BaseModel):
    model_loaded: bool
    model_path: str
    device: str
    parameters: int


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        model_loaded=network is not None,
        model_path=getattr(app.state, 'model_path', ''),
        device=str(device) if device else 'none',
        parameters=sum(p.numel() for p in network.parameters()) if network else 0,
    )


@app.post("/api/move", response_model=MoveResponse)
async def get_move(req: MoveRequest):
    if network is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Input validation ──
    if len(req.board) != 100 or any(len(row) != 100 for row in req.board):
        raise HTTPException(status_code=400, detail="Board must be 100x100")
    if req.current_player not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="current_player must be 1, 2, or 3")
    board_arr = np.array(req.board, dtype=np.int8)
    if not np.all((board_arr >= 0) & (board_arr <= 3)):
        raise HTTPException(status_code=400, detail="Board values must be 0-3")

    import time
    t_start = time.time()

    try:
        env = EightInARowEnv()
        env.board = board_arr
        player_idx = req.current_player - 1
        env.current_player = player_idx
        
        # Manually reconstruct _player_planes since step() wasn't called
        env._player_planes[0] = (env.board == 1).astype(np.float32)
        env._player_planes[1] = (env.board == 2).astype(np.float32)
        env._player_planes[2] = (env.board == 3).astype(np.float32)
        env._player_planes[3] = (env.board == 0).astype(np.float32)
        env._planes_dirty = True
        
        occupied = np.argwhere(env.board != 0)
        if len(occupied) > 0:
            env._min_r = int(occupied[:, 0].min())
            env._max_r = int(occupied[:, 0].max())
            env._min_c = int(occupied[:, 1].min())
            env._max_c = int(occupied[:, 1].max())

        obs, center = env.get_observation(config.local_view_size)
        legal_mask = env.get_legal_actions_mask(center[0], center[1], config.local_view_size)
        if legal_mask.sum() == 0:
            raise HTTPException(status_code=400, detail="No legal moves available")

        network.eval()
        action_probs, _root_value, _root = gumbel_muzero_search(
            network, obs, legal_mask, config,
            add_noise=False
        )
        action = select_action(action_probs, temperature=0.1)
        board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)

        if not (0 <= board_r < env.BOARD_SIZE and 0 <= board_c < env.BOARD_SIZE):
            legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
            if len(legal) > 0:
                board_r, board_c = legal[np.random.randint(len(legal))]
        if env.board[board_r, board_c] != 0:
            legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
            if len(legal) > 0:
                board_r, board_c = legal[np.random.randint(len(legal))]

        elapsed_ms = (time.time() - t_start) * 1000
        confidence = float(action_probs[action]) if action < len(action_probs) else 0.0
        root_value = _root_value.tolist() if hasattr(_root_value, 'tolist') else list(_root_value)
        top_indices = np.argsort(action_probs)[::-1][:3]
        top_actions = []
        for idx in top_indices:
            r, c = env.action_to_board(int(idx), center[0], center[1], config.local_view_size)
            top_actions.append({"row": int(r), "col": int(c), "prob": float(action_probs[idx])})
        return MoveResponse(
            row=int(board_r),
            col=int(board_c),
            confidence=confidence,
            thinking_time_ms=round(elapsed_ms, 1),
            root_value=root_value,
            top_actions=top_actions,
        )
    except HTTPException:
        raise
    except Exception as e:
        _log.error("get_move failed: %s: %s", type(e).__name__, e)
        try:
            legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
            if len(legal) > 0:
                board_r, board_c = legal[np.random.randint(len(legal))]
                return MoveResponse(
                    row=int(board_r),
                    col=int(board_c),
                    confidence=0.0,
                    thinking_time_ms=round((time.time() - t_start) * 1000, 1),
                    root_value=None,
                    top_actions=None,
                )
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=f"Inference error: {type(e).__name__}")


@app.websocket("/api/ws/move")
async def websocket_move(websocket: WebSocket):
    """WebSocket endpoint for AI moves. Yields progress updates while thinking."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        req = MoveRequest(**data)
        
        if network is None:
            await websocket.send_json({"error": "Model not loaded"})
            return
            
        if len(req.board) != 100 or any(len(row) != 100 for row in req.board):
            await websocket.send_json({"error": "Board must be 100x100"})
            return
        if req.current_player not in (1, 2, 3):
            await websocket.send_json({"error": "current_player must be 1, 2, or 3"})
            return
            
        board_arr = np.array(req.board, dtype=np.int8)
        if not np.all((board_arr >= 0) & (board_arr <= 3)):
            await websocket.send_json({"error": "Board values must be 0-3"})
            return

        import time
        t_start = time.time()

        env = EightInARowEnv()
        env.board = board_arr
        player_idx = req.current_player - 1
        env.current_player = player_idx
        
        # Manually reconstruct _player_planes since step() wasn't called
        env._player_planes[0] = (env.board == 1).astype(np.float32)
        env._player_planes[1] = (env.board == 2).astype(np.float32)
        env._player_planes[2] = (env.board == 3).astype(np.float32)
        env._player_planes[3] = (env.board == 0).astype(np.float32)
        env._planes_dirty = True
        
        occupied = np.argwhere(env.board != 0)
        if len(occupied) > 0:
            env._min_r = int(occupied[:, 0].min())
            env._max_r = int(occupied[:, 0].max())
            env._min_c = int(occupied[:, 1].min())
            env._max_c = int(occupied[:, 1].max())

        obs, center = env.get_observation(config.local_view_size)
        legal_mask = env.get_legal_actions_mask(center[0], center[1], config.local_view_size)
        if legal_mask.sum() == 0:
            await websocket.send_json({"error": "No legal moves available"})
            return

        network.eval()
        
        # Run search in a background thread to allow sending progress
        loop = asyncio.get_running_loop()
        search_task = loop.run_in_executor(
            None,
            lambda: gumbel_muzero_search(network, obs, legal_mask, config, add_noise=False)
        )
        
        async def send_progress():
            start_time = time.time()
            while not search_task.done():
                elapsed = time.time() - start_time
                try:
                    await websocket.send_json({"type": "progress", "message": f"Thinking... {elapsed:.1f}s"})
                except Exception:
                    break
                await asyncio.sleep(0.5)
                
        progress_task = asyncio.create_task(send_progress())
        
        try:
            action_probs, _root_value, _root = await search_task
        finally:
            progress_task.cancel()

        action = select_action(action_probs, temperature=0.1)
        board_r, board_c = env.action_to_board(action, center[0], center[1], config.local_view_size)

        if not (0 <= board_r < env.BOARD_SIZE and 0 <= board_c < env.BOARD_SIZE):
            legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
            if len(legal) > 0:
                board_r, board_c = legal[np.random.randint(len(legal))]
        if env.board[board_r, board_c] != 0:
            legal = env.legal_moves_local(center[0], center[1], config.local_view_size)
            if len(legal) > 0:
                board_r, board_c = legal[np.random.randint(len(legal))]

        elapsed_ms = (time.time() - t_start) * 1000
        confidence = float(action_probs[action]) if action < len(action_probs) else 0.0
        root_value = _root_value.tolist() if hasattr(_root_value, 'tolist') else list(_root_value)
        
        top_indices = np.argsort(action_probs)[::-1][:3]
        top_actions = []
        for idx in top_indices:
            r, c = env.action_to_board(int(idx), center[0], center[1], config.local_view_size)
            top_actions.append({"row": int(r), "col": int(c), "prob": float(action_probs[idx])})

        await websocket.send_json({
            "type": "result",
            "row": int(board_r),
            "col": int(board_c),
            "confidence": confidence,
            "thinking_time_ms": round(elapsed_ms, 1),
            "root_value": root_value,
            "top_actions": top_actions
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        _log.error("WebSocket move failed: %s: %s", type(e).__name__, e)
        try:
            await websocket.send_json({"error": f"Inference error: {type(e).__name__}"})
        except: pass


def load_model(model_path: str, use_cuda: bool = True):
    """Load a trained model from checkpoint. On failure, keeps randomly initialized network."""
    global network, config, device

    config = MuZeroConfig()
    config.num_simulations_start = config.num_simulations_end = config.num_simulations_play
    config.validate()

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    network = MuZeroNetwork.from_config(config).to(device)

    if not safe_under(os.getcwd(), model_path):
        _log.warning("Model path escapes base directory (cwd), skipping load: %s", model_path)
    elif os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['network_state_dict'], strict=False)
                print(f"Loaded model from checkpoint: {model_path}")
            else:
                network.load_state_dict(checkpoint, strict=False)
                print(f"Loaded model weights: {model_path}")
        except Exception as e:
            _log.error("Failed to load model %s: %s: %s. Using random init.", model_path, type(e).__name__, e)
    else:
        _log.warning("Model file not found: %s", model_path)

    network.eval()
    print(f"Device: {device}, Parameters: {sum(p.numel() for p in network.parameters()):,}")


def main():
    parser = argparse.ArgumentParser(description="MuZero AI Server")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to model file')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    args = parser.parse_args()

    app.state.model_path = args.model
    load_model(args.model, use_cuda=not args.cpu)

    print(f"\n🚀 Starting AI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
