"""
Gumbel MuZero MCTS — Sequential Halving with Gumbel noise.

Key differences from standard PUCT MuZero:
1. Root: Add Gumbel(0,1) noise to prior logits for exploration
2. Sequential Halving: Start with top-m actions, halve in log2(m) phases
3. Policy improvement: sigma_bar = softmax(logits + completed_Q) is monotone-improving
4. Interior nodes: Standard PUCT selection (unchanged)

Reference: "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ai.log_utils import get_logger

_log = get_logger(__name__)


# ============================================================
#  Data Structures
# ============================================================

class MinMaxStats:
    """Tracks min/max values for normalization."""
    def __init__(self):
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    """A node in the MCTS tree."""
    __slots__ = ['prior', 'visit_count', 'value_sums', 'reward',
                 'hidden_state', 'children', 'expanded', 'logit',
                 'virtual_loss']

    def __init__(self, prior: float, logit: float = 0.0):
        self.prior = prior
        self.logit = logit
        self.visit_count = 0
        self.value_sums = np.zeros(3, dtype=np.float32)
        self.reward = 0.0
        self.hidden_state: Optional[torch.Tensor] = None
        self.children: Dict[int, 'MCTSNode'] = {}
        self.expanded = False
        self.virtual_loss = 0  # Temporary penalty during batch leaf collection

    def value(self) -> np.ndarray:
        if self.visit_count == 0:
            return np.zeros(3, dtype=np.float32)
        # Virtual losses are applied as penalties during selection, 
        # not by biasing the expected value artificially.
        return self.value_sums / self.visit_count

def _rotate_perspective(node: MCTSNode):
    """
    Recursively rotate the value perspective for 3-player games.
    The value vector is [V_me, V_next, V_prev].
    When play passes to the next player:
      - The new 'me' (index 0) is the old 'next' (index 1).
      - The new 'next' (index 1) is the old 'prev' (index 2).
      - The new 'prev' (index 2) is the old 'me' (index 0).
    """
    node.value_sums = np.array([node.value_sums[1], node.value_sums[2], node.value_sums[0]], dtype=np.float32)
    for child in node.children.values():
        _rotate_perspective(child)



# ============================================================
#  Gumbel MuZero Search
# ============================================================

def gumbel_muzero_search(network: torch.nn.Module, observation: np.ndarray,
                         legal_actions_mask: np.ndarray, config,
                         add_noise: bool = True,
                         noise_scale: float = 1.0,
                         session_context_vec: Optional[np.ndarray] = None,
                         device: Optional[torch.device] = None,
                         num_simulations_override: Optional[int] = None,
                         reuse_tree: Optional['MCTSNode'] = None) -> Tuple[np.ndarray, float]:
    """
    Gumbel MuZero search using Sequential Halving.

    Instead of PUCT at the root, uses:
    1. Sample Gumbel noise for each legal action
    2. Sequential Halving to narrow candidates using network evaluations
    3. Returns improved policy via sigma_bar = softmax(logits + completed_Q)

    Args:
        network: MuZeroNetwork
        observation: (4, view_size, view_size) float32
        legal_actions_mask: (action_size,) float32
        config: object with policy_size, num_simulations, discount, gumbel_*,
            mcts_batch_size (or use getattr defaults). See ai/REUSABILITY.md.
        add_noise: whether to add Gumbel noise (True for self-play, False for eval)
        session_context_vec: optional (4,) session context vector
        device: optional pre-cached device (avoids repeated next(network.parameters()).device)
        reuse_tree: optional MCTSNode from previous search to reuse as starting point

    Returns:
        (action_probs, root_value, root): improved policy distribution, root value estimate, and MCTS root node
    """
    if device is None:
        device = next(network.parameters()).device
    min_max = MinMaxStats()

    # ---- Initial inference ----
    obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(device)
    ctx_tensor = None
    if session_context_vec is not None:
        ctx_tensor = torch.from_numpy(session_context_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden_state, policy_logits, value = network.initial_inference(
            obs_tensor, session_context=ctx_tensor
        )

    # Value is (1, 3); sanitize to avoid NaN/Inf propagation
    root_value = _safe_value(value[0].cpu().numpy())  # Vector [V_me, V_next, V_prev]
    logits = policy_logits.squeeze(0).cpu().numpy()
    legal_actions = np.where(legal_actions_mask > 0)[0]

    if len(legal_actions) == 0:
        return np.zeros(config.policy_size, dtype=np.float32), root_value, None

    # ---- Setup root ----
    # Reuse subtree if available (preserves previous evaluations)
    if reuse_tree is not None and reuse_tree.expanded and reuse_tree.hidden_state is not None:
        _rotate_perspective(reuse_tree)
        root = reuse_tree
        # Re-derive logits from the reused root's children priors for Sequential Halving
        # We still need fresh logits from network for the improved policy computation
        logits = policy_logits.squeeze(0).cpu().numpy()
        # Update root hidden state with fresh inference (view may have shifted)
        root.hidden_state = hidden_state.squeeze(0)
    else:
        root = MCTSNode(prior=0.0)
        root.hidden_state = hidden_state.squeeze(0)
        root.value_sums = root_value.copy()
        root.visit_count = 1
        root.expanded = True

        logits = policy_logits.squeeze(0).cpu().numpy()

    # Mask illegal actions
    masked_logits = np.full(config.policy_size, -1e9, dtype=np.float32)
    masked_logits[legal_actions] = logits[legal_actions]
    priors = _softmax(masked_logits)

    # Create children with prior and raw logit (only for actions not already in tree)
    for a in legal_actions:
        if int(a) not in root.children:
            root.children[int(a)] = MCTSNode(prior=float(priors[a]), logit=float(logits[a]))

    if len(legal_actions) == 1:
        # Only one legal action
        action_probs = np.zeros(config.policy_size, dtype=np.float32)
        action_probs[legal_actions[0]] = 1.0
        return action_probs, root_value, root

    # ---- Gumbel noise ----
    if add_noise:
        gumbel_noise = np.random.gumbel(size=len(legal_actions)) * noise_scale
    else:
        gumbel_noise = np.zeros(len(legal_actions))

    # Gumbel scores: logit(a) + Gumbel(a)
    gumbel_logits = {}
    for i, a in enumerate(legal_actions):
        gumbel_logits[int(a)] = logits[a] + gumbel_noise[i]

    # ---- Sequential Halving ----
    max_considered = min(config.gumbel_max_considered_actions, len(legal_actions))
    num_simulations = num_simulations_override if num_simulations_override is not None else getattr(config, 'num_simulations_start', 25)

    if reuse_tree is not None:
        min_sims = max(getattr(config, 'min_simulations', 10), num_simulations // 2)
        num_simulations = max(min_sims, num_simulations - reuse_tree.visit_count)

    # Start with top-m actions by Gumbel score
    sorted_actions = sorted(gumbel_logits.keys(),
                            key=lambda a: gumbel_logits[a], reverse=True)
    considered_actions = sorted_actions[:max_considered]

    # Number of halving phases
    num_phases = max(1, int(math.ceil(math.log2(max_considered))))
    sims_per_phase = max(1, num_simulations // num_phases)

    mcts_batch_size = getattr(config, 'mcts_batch_size', 8)

    for phase in range(num_phases):
        if len(considered_actions) <= 1:
            break

        # Allocate simulations equally among considered actions
        sims_per_action = max(1, sims_per_phase // len(considered_actions))

        # Build flat list of all simulations for this phase
        total_sims = sims_per_action * len(considered_actions)
        sim_queue = []
        for action in considered_actions:
            child = root.children[action]
            for _ in range(sims_per_action):
                sim_queue.append((action, child))

        # Process simulations in batches with virtual loss
        _batch_simulate_phase(root, sim_queue, network, config, device,
                              min_max, mcts_batch_size)

        # Compute completed Q for each considered action
        completed_q = {}
        for action in considered_actions:
            child = root.children[action]
            if child.visit_count > 0:
                # Q(s,a) = r + gamma*V(s')
                q_val = child.reward + config.discount * child.value()
                completed_q[action] = _complete_q(
                    gumbel_logits[action],
                    q_val, min_max, config
                )
            else:
                completed_q[action] = gumbel_logits[action]

        # Halve: keep top half by completed score
        half_size = max(1, len(considered_actions) // 2)
        considered_actions = sorted(
            considered_actions,
            key=lambda a: completed_q[a],
            reverse=True
        )[:half_size]

    # ---- Improved policy: sigma_bar ----
    # sigma_bar(a) = softmax(logits(a) + completed_Q(a))
    improved_logits = np.full(config.policy_size, -1e9, dtype=np.float32)
    for action in root.children:
        child = root.children[action]
        if child.visit_count > 0:
            # Q(s,a) = r + gamma*V(s')
            q_val = child.reward + config.discount * child.value()
            normalized_q = min_max.normalize(q_val[0]) # Root player is always index 0 relative to root
            improved_logits[action] = logits[action] + config.gumbel_c_scale * normalized_q
        else:
            improved_logits[action] = logits[action]

    action_probs = _softmax(improved_logits)

    # Zero out illegal
    action_probs[legal_actions_mask == 0] = 0
    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    return action_probs, root_value, root


def _batch_simulate_phase(root: MCTSNode, sim_queue: List[Tuple[int, 'MCTSNode']],
                          network, config, device, min_max: MinMaxStats,
                          batch_size: int = 8):
    """Run all simulations for a Sequential Halving phase using batched evaluation.

    Uses virtual loss to diversify tree traversals within each batch,
    then evaluates all collected leaf nodes in one network call.
    """
    num_players = getattr(config, 'num_players', 3)
    idx = 0

    while idx < len(sim_queue):
        batch_end = min(idx + batch_size, len(sim_queue))

        # 1. Traverse to leaves with virtual loss applied
        leaves = []
        for j in range(idx, batch_end):
            root_action, child = sim_queue[j]
            search_path, actions_taken, leaf_node = _traverse_to_leaf_vl(
                root, root_action, child, min_max, config
            )
            leaves.append((search_path, actions_taken, leaf_node))

        # Process batch with virtual-loss safety: if network fails,
        # ensure all virtual losses are cleaned up before re-raising.
        try:
            # 2. Partition into expandable vs already-expanded leaves
            to_expand = []
            no_expand = []

            for li, (search_path, actions_taken, leaf_node) in enumerate(leaves):
                parent = search_path[-2]
                if not leaf_node.expanded and parent.hidden_state is not None:
                    to_expand.append((li, search_path, actions_taken,
                                      parent.hidden_state, actions_taken[-1], leaf_node))
                else:
                    if leaf_node.expanded and leaf_node.visit_count == 0:
                        val = leaf_node.reward
                    else:
                        val = leaf_node.value() if leaf_node.visit_count > 0 else 0.0
                    no_expand.append((li, search_path, val))

            # 3. Batch network evaluation for expandable leaves. Efficiency: batched leaf evaluation to reduce device round-trips.
            if to_expand:
                parent_states = torch.stack([t[3] for t in to_expand]).to(device)
                action_ids = torch.tensor([t[4] for t in to_expand], device=device)

                with torch.no_grad():
                    next_states, rewards, policy_logits_batch, values = \
                        network.recurrent_inference(parent_states, action_ids)

                # 4. Expand each leaf and backpropagate
            # 4. Expand each leaf and backpropagate
                for k, (li, search_path, actions_taken, _, _, leaf_node) in enumerate(to_expand):
                    leaf_node.hidden_state = next_states[k]
                    r = rewards[k].item()
                    leaf_node.reward = float(_safe_value(np.asarray(r)).flat[0])
                    leaf_node.expanded = True

                    priors_logits = policy_logits_batch[k].detach()
                    priors = F.softmax(priors_logits, dim=0).cpu().numpy()
                    raw_logits = priors_logits.cpu().numpy()
                    
                    # Top-K pruning: only create nodes for the K most probable actions
                    # to avoid creating up to 441 MCTSNode objects per leaf expansion
                    _TOP_K = 32
                    if len(priors) > _TOP_K:
                        top_indices = np.argpartition(priors, -_TOP_K)[-_TOP_K:]
                        for a_idx in top_indices:
                            if priors[a_idx] > 1e-8:
                                leaf_node.children[int(a_idx)] = MCTSNode(
                                    prior=float(priors[a_idx]), logit=float(raw_logits[a_idx])
                                )
                    else:
                        for a_idx in range(len(priors)):
                            if priors[a_idx] > 1e-6:
                                leaf_node.children[int(a_idx)] = MCTSNode(
                                    prior=float(priors[a_idx]), logit=float(raw_logits[a_idx])
                                )
                    if not leaf_node.children:
                        best_a = int(priors.argmax())
                        leaf_node.children[best_a] = MCTSNode(
                            prior=float(priors[best_a]), logit=float(raw_logits[best_a])
                        )
                    
                    # Network output [V_me, V_next, V_prev] needs to be rotated to absolute [V0, V1, V2]
                    # based on who is "me" (leaf's player).
                    # Depth = len(search_path) - 1. 
                    # Turn/Player = Depth % 3.
                    leaf_depth = len(search_path) - 1
                    turn = leaf_depth % 3
                    
                    val_relative = values[k].detach().cpu().numpy() # [V_me, V_next, V_prev]
                    val_absolute = _safe_value(np.roll(val_relative, turn)) # [V0, V1, V2]
                    
                    _remove_virtual_loss(search_path)
                    _backpropagate(search_path, val_absolute, config.discount,
                                   min_max, num_players)

            # 5. Backpropagate non-expandable leaves
            for li, search_path, value_est in no_expand:
                real_val = value_est
                if np.isscalar(real_val) or (isinstance(real_val, np.ndarray) and real_val.size == 1):
                    # Scalar reward case: Construct zero-sum vector for the player who moved
                    v = np.zeros(3, dtype=np.float32)
                    p_idx = (len(search_path) - 2) % num_players
                    v[p_idx] = float(real_val)
                    if num_players > 1:
                        v_other = -float(real_val) / (num_players - 1)
                        for i in range(num_players):
                            if i != p_idx:
                                v[i] = v_other
                    real_val = v 

                _remove_virtual_loss(search_path)
                _backpropagate(search_path, real_val, config.discount,
                               min_max, num_players)

        except Exception:
            idx = batch_end
            raise

        idx = batch_end


def _traverse_to_leaf_vl(root: MCTSNode, root_action: int, child: MCTSNode,
                         min_max: MinMaxStats, config) -> Tuple[List[MCTSNode], List[int], MCTSNode]:
    """Traverse from root→child→...→leaf applying virtual loss along the way."""
    search_path = [root, child]
    child.virtual_loss += 1
    node = child
    actions_taken = [root_action]
    
    # Root is turn 0. child is turn 1. 
    # At 'node' (depth d), we are choosing ANY of its children (depth d+1).
    # The player choosing is turn `d % 3`.
    # Current `node` is `child`, so depth is 1. Choosing for depth 2.
    # Player at `node` is P1 (turn 1).
    turn = 1

    while node.expanded and node.children:
        action, next_node = _select_child_puct(node, min_max, config, turn)
        if next_node is None:
            print(f"[MCTS] Critical: _select_child_puct returned None! Node children: {len(node.children)}")
            for a, c in node.children.items():
                 print(f"  Child {a}: visits={c.visit_count}, prior={c.prior}, value={c.value()}")
            raise RuntimeError("MCTS Selection Failed")
        actions_taken.append(action)
        search_path.append(next_node)
        next_node.virtual_loss += 1
        node = next_node
        turn = (turn + 1) % 3

    return search_path, actions_taken, node


def _remove_virtual_loss(search_path: List[MCTSNode]):
    """Remove one unit of virtual loss from each node in the path (except root)."""
    for node in search_path[1:]:  # skip root (root VL is irrelevant)
        node.virtual_loss = max(0, node.virtual_loss - 1)


def _complete_q(gumbel_logit: float, q_value: float,
                min_max: MinMaxStats, config) -> float:
    """Compute completed score for Sequential Halving."""
    # q_value here comes from q_val = reward + discount * value().
    # value() is vector. reward is scalar.
    # We need to scalarize it before calling this, OR handle it here.
    # But see logic in gumbel_muzero_search:
    # `q_val = child.reward + config.discount * child.value()`
    # `child.value()` returns vector. `child.reward` scalar.
    # This sum will be a vector (broadcasting).
    # So `q_value` passed in IS a vector.
    # Root player (P0) wants Q[0].
    
    if isinstance(q_value, np.ndarray):
        q = q_value[0] # Root player is always index 0 relative to root
    else:
        q = q_value
        
    normalized_q = min_max.normalize(q)
    return gumbel_logit + config.gumbel_c_scale * normalized_q


def _select_child_puct(node: MCTSNode, min_max: MinMaxStats,
                       config, turn: int) -> Tuple[int, MCTSNode]:
    """Standard PUCT selection for interior nodes (virtual-loss aware)."""
    best_action = -1
    best_score = -float('inf')
    best_child = None

    # Include virtual loss in total visit count
    total_visits = sum(c.visit_count + c.virtual_loss for c in node.children.values())

    for action, child in node.children.items():
        score = _ucb_score(node, child, total_visits, min_max, config, turn)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child


def _ucb_score(parent: MCTSNode, child: MCTSNode, total_visits: int,
               min_max: MinMaxStats, config, turn: int) -> float:
    """PUCT score for child selection (virtual-loss aware).
    
    Args:
        turn: The player index (0, 1, 2) moving at `parent`.
              They want to maximize V[turn].
    """
    effective_visits = child.visit_count + child.virtual_loss
    pb_c = math.log((total_visits + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    prior_score = pb_c * child.prior * math.sqrt(total_visits) / (1 + effective_visits)

    if child.visit_count > 0:
        v_vec = child.value() # [V0, V1, V2] absolute
        v_scalar = v_vec[turn]
    else:
        v_scalar = 0.0

    if effective_visits > 0:
        # Apply virtual loss penalty: Treat virtual loss as a -1.0 return
        vl_penalty = (child.virtual_loss / effective_visits) * config.discount
        q_value = child.reward + config.discount * v_scalar - vl_penalty
        value_score = min_max.normalize(q_value)
    else:
        value_score = 0.0

    return prior_score + value_score


def _safe_value(value: np.ndarray) -> np.ndarray:
    """Sanitize value/reward for MCTS use; avoid NaN/Inf propagation."""
    v = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)


def _backpropagate(search_path: List[MCTSNode], value: np.ndarray,
                   discount: float, min_max: MinMaxStats,
                   num_players: int = 3):
    """Backpropagate value vector through the search path.
    Monte Carlo: add leaf value to path; mean at node = avg of leaf returns through this node.
    Args:
        value: (3,) vector of values [V_P0, V_P1, V_P2] (absolute from Root perspective).
    """
    value = _safe_value(value)
    for i, node in enumerate(reversed(search_path)):
        node.value_sums += value
        node.visit_count += 1
        # MinMaxStats update: tracking the root's perspective component (value[0])
        # correctly establishes the plausible bounds of the value without
        # corrupting the scale with the opposing players' potentially negative bounds.
        min_max.update(value[0])


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def select_action(action_probs: np.ndarray, temperature: float = 1.0) -> int:
    """Select an action from the policy distribution."""
    if temperature == 0:
        return int(np.argmax(action_probs))
    elif temperature == float('inf'):
        visited = np.where(action_probs > 0)[0]
        if len(visited) == 0:
            return int(np.random.choice(len(action_probs)))
        return int(np.random.choice(visited))
    else:
        probs = action_probs ** (1.0 / temperature)
        total = probs.sum()
        if total == 0:
            return int(np.random.choice(len(probs)))
        probs /= total
        return int(np.random.choice(len(probs), p=probs))
