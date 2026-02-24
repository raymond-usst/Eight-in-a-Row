"""Population-based training: population of agents with evolutionary hyperparameter updates.

Maintainability: load/save retry — see ai/MAINTENANCE.md.
"""

import os
import torch
import random
import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from ai.muzero_config import MuZeroConfig
from ai.log_utils import get_logger

_log = get_logger(__name__)


@dataclass
class PBTAgent:
    """Represents a single agent in the PBT population."""
    agent_id: int
    config: MuZeroConfig
    model_state: Dict[str, Any] = field(default_factory=dict)
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    
    # PBT Hyperparameters (that can be mutated)
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Metadata
    win_rate: float = 0.0
    games_played: int = 0
    generation: int = 0
    parent_id: int = -1  # ID of agent copied from (if any)
    
    def get_hyperparams(self) -> Dict[str, float]:
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }

    def set_hyperparams(self, params: Dict[str, float]):
        self.learning_rate = params.get('learning_rate', self.learning_rate)
        self.weight_decay = params.get('weight_decay', self.weight_decay)

class Population:
    """Manages the population of agents and evolutionary logic."""
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.size = config.pbt_population_size
        self.agents: List[PBTAgent] = []
        self.generation = 0
        
        # Initialize population
        for i in range(self.size):
            agent = PBTAgent(
                agent_id=i,
                config=copy.deepcopy(config)
            )
            # Initialize with random perturbations if needed?
            # For now, start identical (or rely on random init if fresh)
            self.agents.append(agent)

    def update_agent_stats(self, agent_id: int, win_rate: float, games: int):
        if 0 <= agent_id < self.size:
            self.agents[agent_id].win_rate = win_rate
            self.agents[agent_id].games_played = games

    def _recursive_to_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self._recursive_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_to_cpu(v) for v in obj]
        else:
            return obj

    def sync_agent_weights(self, agent_id: int, model, optimizer):
        """Update agent's stored weights from the active running model. Uses unwrapped module if compiled so keys are plain (no _orig_mod.)."""
        if 0 <= agent_id < self.size:
            target = getattr(model, '_orig_mod', model)
            self.agents[agent_id].model_state = {k: v.cpu().clone() for k, v in target.state_dict().items()}
            self.agents[agent_id].optimizer_state = self._recursive_to_cpu(optimizer.state_dict())
            
            # Sync current hyperparams from agent to optimizer? No, usually other way around.
            # Actually, PBT dictates the hyperparams. The optimizer follows the agent.

    def load_agent_weights(self, agent_id: int, model, optimizer):
        """Load agent's weights into the running model. Loads into unwrapped module if compiled (agent state is plain keys)."""
        if 0 <= agent_id < self.size:
            agent = self.agents[agent_id]
            if agent.model_state:
                target = getattr(model, '_orig_mod', model)
                target.load_state_dict(agent.model_state, strict=False)
            if agent.optimizer_state:
                optimizer.load_state_dict(agent.optimizer_state)
            
            # Apply PBT hyperparams to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = agent.learning_rate
                param_group['weight_decay'] = agent.weight_decay

    def exploit_and_explore(self):
        """
        PBT Step:
        1. Rank agents by win rate.
        2. Replace bottom 25% with copies of top 25% (Exploit).
        3. Mutate hyperparams of the copies (Explore).
        """
        # Sort by win rate (descending)
        # Filter agents with 0 games? Maybe not, allow random chance?
        # Better to sort all.
        sorted_indices = sorted(range(self.size), key=lambda i: self.agents[i].win_rate, reverse=True)
        
        num_replace = max(1, int(self.size * 0.25))
        top_indices = sorted_indices[:num_replace]
        bottom_indices = sorted_indices[-num_replace:]
        
        print(f"[PBT] Evolution Gen {self.generation}: Top {top_indices} -> Bottom {bottom_indices}")
        
        for i, bottom_idx in enumerate(bottom_indices):
            top_idx = top_indices[i % len(top_indices)]
            
            source = self.agents[top_idx]
            target = self.agents[bottom_idx]
            
            # Exploit: Copy weights and hyperparams
            target.model_state = copy.deepcopy(source.model_state)
            target.optimizer_state = copy.deepcopy(source.optimizer_state)
            target.learning_rate = source.learning_rate
            target.weight_decay = source.weight_decay
            target.parent_id = source.agent_id
            target.generation = self.generation
            
            # Explore: Mutate
            if random.random() < self.config.pbt_mutation_rate:
                # Perturb LR by factor of 1.2 or 0.8
                factor = 1.2 if random.random() > 0.5 else 0.8
                target.learning_rate *= factor
                print(f"  - Agent {bottom_idx} mutated LR: {source.learning_rate:.2e} -> {target.learning_rate:.2e}")

            if random.random() < self.config.pbt_mutation_rate:
                # Perturb Weight Decay
                factor = 1.2 if random.random() > 0.5 else 0.8
                target.weight_decay *= factor
                print(f"  - Agent {bottom_idx} mutated WD: {source.weight_decay:.2e} -> {target.weight_decay:.2e}")
            
            # Reset stats
            target.win_rate = 0.0
            target.games_played = 0
            
        self.generation += 1

    def save(self, path: str):
        payload = {
            'generation': self.generation,
            'agents': [
                {
                    'id': a.agent_id,
                    'model': a.model_state,
                    'opt': a.optimizer_state,
                    'hyperparams': a.get_hyperparams(),
                    'stats': {'win_rate': a.win_rate, 'games': a.games_played},
                    'meta': {'gen': a.generation, 'parent': a.parent_id}
                }
                for a in self.agents
            ]
        }
        torch.save(payload, path)
    
    def load(self, path: str):
        if not os.path.exists(path):
            return
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            self.generation = payload.get('generation', 0)
            agent_data = payload.get('agents', [])
            if not isinstance(agent_data, list):
                agent_data = []
            skipped = 0
            for data in agent_data:
                if not isinstance(data, dict):
                    skipped += 1
                    continue
                aid = data.get('id', data.get('agent_id'))
                if aid is None or not (0 <= aid < self.size):
                    skipped += 1
                    continue
                model = data.get('model', None)
                opt = data.get('opt', None)
                hyperparams = data.get('hyperparams', None)
                stats = data.get('stats')
                meta = data.get('meta') or {}
                if model is None or opt is None:
                    skipped += 1
                    continue
                agent = self.agents[aid]
                agent.model_state = model
                agent.optimizer_state = opt
                agent.set_hyperparams(hyperparams if isinstance(hyperparams, dict) else {})
                agent.win_rate = stats.get('win_rate', 0.0) if isinstance(stats, dict) else 0.0
                agent.games_played = stats.get('games', 0) if isinstance(stats, dict) else 0
                agent.generation = meta.get('gen', 0) if isinstance(meta, dict) else 0
                agent.parent_id = meta.get('parent', -1) if isinstance(meta, dict) else -1
            if skipped > 0:
                _log.warning("Skipped %d agent(s) with missing or invalid keys.", skipped)
            print(f"[PBT] Loaded population (Gen {self.generation}) from {path}")
        except Exception as e:
            _log.error("Failed to load population: %s", e)
