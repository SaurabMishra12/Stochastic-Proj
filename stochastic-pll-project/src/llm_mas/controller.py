"""A fixed stochastic communication policy, inspired by stochastic feedback.

The scalar state is an OU-style controller variable, not a physical phase or a
learned latent representation. Three Euler--Maruyama substeps schedule the two
candidate outgoing edges of senders 0, 1 and 2, respectively. No agent response
changes during these numerical substeps.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random


@dataclass(frozen=True)
class ControllerConfig:
    step_size: float = 0.1
    mean_reversion: float = 1.2
    noise_scale: float = 0.35
    initial_state: float = 0.0
    state_min: float = -2.0
    state_max: float = 2.0
    logit_offset: float = -1.25
    disagreement_gain: float = 2.0
    state_gain: float = 0.8

    def __post_init__(self):
        values = asdict(self)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
            raise ValueError('Controller parameters must be finite numbers.')
        if self.step_size <= 0 or self.mean_reversion <= 0 or self.noise_scale <= 0:
            raise ValueError('Controller step, reversion and noise must be positive.')
        if not self.state_min < self.state_max or not self.state_min <= self.initial_state <= self.state_max:
            raise ValueError('Invalid controller bounds or initial state.')
        if self.disagreement_gain <= 0 or self.state_gain <= 0:
            raise ValueError('Controller feedback gains must be positive.')
        extreme_logits = [self.logit_offset + self.disagreement_gain * d + self.state_gain * x
                          for d in (0, 1) for x in (self.state_min, self.state_max)]
        if max(map(abs, extreme_logits)) > 30:
            raise ValueError('Controller configuration would degenerate to an effectively fixed gate.')


def controller_seed(seed: int, condition: str, item_id: str, replicate_id: int) -> int:
    """Create a process-independent stream identity without Python's hash()."""
    identity = json.dumps(['mas-controller-v1', seed, condition, str(item_id), replicate_id],
                          ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:16], 'big')


def stochastic_edges(disagreement: float, item_id: str, seed: int,
                     replicate_id: int, config: ControllerConfig) -> dict:
    """Return edges and all random quantities needed to replay the routing."""
    if not math.isfinite(disagreement) or not 0 <= disagreement <= 1:
        raise ValueError('Disagreement must be in [0, 1].')
    stream_seed = controller_seed(seed, 'stochastic_controller', item_id, replicate_id)
    rng = random.Random(stream_seed)
    state = config.initial_state
    result = {'edges': [], 'controller_seed': stream_seed, 'states': [],
              'edge_probabilities': [], 'uniform_draws': []}
    h = config.step_size
    for sender in range(3):
        innovation = rng.gauss(0.0, 1.0)
        previous = state
        drift = config.mean_reversion * (disagreement - previous)
        proposed = previous + drift * h + config.noise_scale * math.sqrt(h) * innovation
        state = min(config.state_max, max(config.state_min, proposed))
        logit = config.logit_offset + config.disagreement_gain * disagreement + config.state_gain * state
        probability = 1.0 / (1.0 + math.exp(-logit))
        result['states'].append({'substep': sender, 'time': (sender + 1) * h,
                                 'previous_state': previous, 'target': disagreement,
                                 'drift': drift, 'gaussian_draw': innovation,
                                 'proposed_state': proposed, 'state': state,
                                 'probability': probability})
        for receiver in range(3):
            if sender == receiver:
                continue
            edge = [sender, receiver]
            draw = rng.random()
            result['edge_probabilities'].append({'edge': edge, 'probability': probability})
            result['uniform_draws'].append({'edge': edge, 'draw': draw})
            if draw < probability:
                result['edges'].append(edge)
    return result
