import zmq
import torch
import numpy as np
import threading
import time
from typing import Dict, Tuple, Optional, Any
from ai.log_utils import get_logger

_log = get_logger(__name__)

# Magic header bytes to distinguish method calls
OP_INITIAL = b'\x01'
OP_RECURRENT = b'\x02'
OP_CENTER = b'\x03'

class InferenceClient:
    """Client for Batched IPC Inference. Acts as a drop-in replacement for MuZeroNetwork inside self_play.py."""
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(endpoint)
        self.endpoint = endpoint
        # The actor process using this object must use it synchronously.
        # ZeroMQ REQ sockets block until reply is received.

    def _send_and_recv(self, op: bytes, tensors: list) -> list:
        # Build multipart message: [OP, shape(s), data(s)]
        # For simplicity, we assume shapes are fixed/known or we send them as meta.
        # Better: send shapes using numpy header or custom packed header.
        # But for zero-copy, we want to just send raw bytes.
        parts = [op]
        for arr in tensors:
            if arr is not None:
                # Need to convert boolean to uint8 for zero-copy
                if arr.dtype == bool:
                    arr = arr.astype(np.uint8)
                meta = f"{arr.dtype.name}:{','.join(map(str, arr.shape))}".encode('utf-8')
                parts.append(meta)
                parts.append(arr)
            else:
                parts.append(b"NONE")

        self.socket.send_multipart(parts)
        
        reply_parts = self.socket.recv_multipart()
        
        # Parse replies
        results = []
        i = 0
        while i < len(reply_parts):
            meta = reply_parts[i]
            if meta == b"NONE":
                results.append(None)
                i += 1
            else:
                dtype_str, shape_str = meta.decode('utf-8').split(':')
                shape = tuple(map(int, shape_str.split(','))) if shape_str else ()
                data_bytes = reply_parts[i+1]
                
                arr = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str))
                if shape:
                    arr = arr.reshape(shape)
                
                # Convert back to torch tensor, but keep on CPU since actor loop runs on CPU
                results.append(torch.from_numpy(arr.copy()))  # copy to avoid read-only buffer issues
                i += 2
                
        return results

    def initial_inference(self, observation: torch.Tensor, session_context: Optional[torch.Tensor] = None):
        """Replacement for MuZeroNetwork.initial_inference"""
        # obs is usually (1, 8, 21, 21) on GPU during training, but here it's on CPU.
        # It's an array of float32.
        obs_np = observation.numpy() if isinstance(observation, torch.Tensor) else observation
        ctx_np = session_context.numpy() if session_context is not None else None
        
        results = self._send_and_recv(OP_INITIAL, [obs_np, ctx_np])
        hidden_state, policy_logits, value = results[0], results[1], results[2]
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor):
        """Replacement for MuZeroNetwork.recurrent_inference"""
        h_np = hidden_state.numpy() if isinstance(hidden_state, torch.Tensor) else hidden_state
        a_np = action.numpy() if isinstance(action, torch.Tensor) else action
        
        results = self._send_and_recv(OP_RECURRENT, [h_np, a_np])
        next_hidden, reward, policy_logits, value = results[0], results[1], results[2], results[3]
        return next_hidden, reward, policy_logits, value

    def predict_center(self, global_state: torch.Tensor):
        """Replacement for FocusNetwork.predict_center"""
        gs_np = global_state.numpy() if isinstance(global_state, torch.Tensor) else global_state
        
        results = self._send_and_recv(OP_CENTER, [gs_np])
        cr, cc = int(results[0].item()), int(results[1].item())
        return cr, cc
        
    def parameters(self):
        """Mock parameters method to keep self_play.py logic happy"""
        # Return a mock tensor on CPU so `device = next(network.parameters()).device` yields CPU
        yield torch.tensor(0.0, device='cpu')

    def eval(self):
        pass


class InferenceServer:
    """Server for IPC Batched Inference. 
    Runs on the main GPU orchestrator machine or a dedicated process.
    Listens on a ZeroMQ ROUTER socket, batches requests, and calls the PyTorch models."""
    def __init__(self, model: torch.nn.Module, config, device: torch.device, endpoint: str = "tcp://0.0.0.1:5555"):
        self.model = model
        self.config = config
        self.device = device
        self.endpoint = endpoint
        self.batch_size = config.ipc_batch_size
        self.timeout_ms = config.ipc_timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(endpoint)
        self.running = False
        
    def _parse_tensors(self, parts: list):
        """Parses [meta, data, meta, data...] into numpy arrays."""
        tensors = []
        i = 0
        while i < len(parts):
            meta = parts[i]
            if meta == b"NONE":
                tensors.append(None)
                i += 1
            else:
                dtype_str, shape_str = meta.decode('utf-8').split(':')
                shape = tuple(map(int, shape_str.split(','))) if shape_str else ()
                data_bytes = parts[i+1]
                arr = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str))
                if shape:
                    arr = arr.reshape(shape)
                tensors.append(arr)
                i += 2
        return tensors

    def _serialize_tensors(self, tensors: list) -> list:
        """Serializes list of tensors/arrays into [meta, data, meta, data...]."""
        parts = []
        for t in tensors:
            if t is None:
                parts.append(b"NONE")
            else:
                arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
                if arr.dtype == bool:
                    arr = arr.astype(np.uint8)
                meta = f"{arr.dtype.name}:{','.join(map(str, arr.shape))}".encode('utf-8')
                parts.append(meta)
                # Important: must be contiguous and viewable as bytes
                if not arr.flags['C_CONTIGUOUS']:
                    arr = np.ascontiguousarray(arr)
                parts.append(arr)
        return parts

    def serve_forever(self):
        self.running = True
        self.model.eval()
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        _log.info(f"IPC Inference Server started at {self.endpoint} (batch={self.batch_size}, timeout={self.timeout_ms}ms)")
        
        total_batches = 0
        total_items = 0
        last_log_time = time.time()
        
        while self.running:
            # Check for incoming requests
            pending = {}
            # Wait up to timeout_ms for the first request
            events = poller.poll(timeout=100) # 100ms blocks if completely idle
            
            if not events:
                continue
                
            # Got at least 1 event. Now poll aggressively until batch_size or timeout.
            t_start = time.time()
            
            while len(pending) < self.batch_size:
                try:
                    # Non-blocking receive for subsequent messages
                    msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                    client_addr = msg[0]
                    empty = msg[1] # standard REQ/ROUTER delim
                    op = msg[2]
                    payload = msg[3:]
                    
                    if op not in pending:
                        pending[op] = []
                        
                    pending[op].append((client_addr, payload))
                except zmq.Again:
                    # Queue is empty, check time
                    if (time.time() - t_start) * 1000 > self.timeout_ms:
                        break
                    time.sleep(0.0001) # 0.1ms yield
                    
            if not pending:
                continue
                
            # Process batches by OP type. Group by tensor shape so curriculum (different board sizes) never concatenate.
            def group_by_shape(addrs, payloads, key_idx=0):
                """Group (addrs, payloads) by shape of payload[key_idx]. Returns list of (addrs_sub, payloads_sub)."""
                groups = {}
                for addr, p in zip(addrs, payloads):
                    sh = p[key_idx].shape
                    if sh not in groups:
                        groups[sh] = ([], [])
                    groups[sh][0].append(addr)
                    groups[sh][1].append(p)
                return list(groups.values())
            
            with torch.no_grad():
                for op, reqs in pending.items():
                    addrs = [req[0] for req in reqs]
                    payloads = [self._parse_tensors(req[1]) for req in reqs]
                    
                    # Tracking batch sizes
                    total_batches += 1
                    total_items += sum([p[0].shape[0] for p in payloads])
                    
                    if time.time() - last_log_time > 10.0:
                        _log.info(f"IPC Avg Batch Size: {total_items / max(1, total_batches):.1f}")
                        total_batches = 0
                        total_items = 0
                        last_log_time = time.time()
                    
                    if op == OP_INITIAL:
                        # payloads are [[obs, ctx], ...]; group by obs shape (board size)
                        for addrs_g, payloads_g in group_by_shape(addrs, payloads, 0):
                            obs_list = [p[0] for p in payloads_g]
                            ctx_list = [p[1] for p in payloads_g]
                            obs_tensor = torch.from_numpy(np.concatenate(obs_list, axis=0)).to(self.device)
                            if ctx_list[0] is not None:
                                ctx_tensor = torch.from_numpy(np.concatenate(ctx_list, axis=0)).to(self.device)
                            else:
                                ctx_tensor = None
                            hidden_states, policy_logits, values = self.model.initial_inference(obs_tensor, session_context=ctx_tensor)
                            offset = 0
                            for i, addr in enumerate(addrs_g):
                                bs = payloads_g[i][0].shape[0]
                                res = [hidden_states[offset:offset+bs], policy_logits[offset:offset+bs], values[offset:offset+bs]]
                                self.socket.send_multipart([addr, b""] + self._serialize_tensors(res))
                                offset += bs
                            
                    elif op == OP_RECURRENT:
                        # hidden/action shapes are typically uniform; group by hidden shape for safety
                        for addrs_g, payloads_g in group_by_shape(addrs, payloads, 0):
                            h_list = [p[0] for p in payloads_g]
                            a_list = [p[1] for p in payloads_g]
                            h_tensor = torch.from_numpy(np.concatenate(h_list, axis=0)).to(self.device)
                            a_tensor = torch.from_numpy(np.concatenate(a_list, axis=0)).to(self.device)
                            next_hiddens, rewards, policy_logits, values = self.model.recurrent_inference(h_tensor, a_tensor)
                            offset = 0
                            for i, addr in enumerate(addrs_g):
                                bs = payloads_g[i][0].shape[0]
                                res = [next_hiddens[offset:offset+bs], rewards[offset:offset+bs], policy_logits[offset:offset+bs], values[offset:offset+bs]]
                                self.socket.send_multipart([addr, b""] + self._serialize_tensors(res))
                                offset += bs
                            
                    elif op == OP_CENTER:
                        # Focus network: group by global_state shape (board size)
                        for addrs_g, payloads_g in group_by_shape(addrs, payloads, 0):
                            gs_list = [p[0] for p in payloads_g]
                            gs_tensor = torch.from_numpy(np.concatenate(gs_list, axis=0)).to(self.device)
                            if hasattr(self.model, 'focus_net'):
                                coords = self.model.focus_net(gs_tensor)
                                w = gs_tensor.shape[-1]
                                for i, addr in enumerate(addrs_g):
                                    r = int(coords[i, 0].item() * w)
                                    c = int(coords[i, 1].item() * w)
                                    r = max(0, min(w-1, r))
                                    c = max(0, min(w-1, c))
                                    res = [torch.tensor([r], dtype=torch.int32), torch.tensor([c], dtype=torch.int32)]
                                    self.socket.send_multipart([addr, b""] + self._serialize_tensors(res))
                            else:
                                for addr in addrs_g:
                                    res = [torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)]
                                    self.socket.send_multipart([addr, b""] + self._serialize_tensors(res))
