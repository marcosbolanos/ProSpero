from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from prospero.search.puct import ActionPrior, PUCTConfig, PUCTSearch


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ProSSTAction:
    aa: str
    log_delta: float
    full_vocab_logp: float
    logp_sampled: float
    logp_original: float
    prior: float


@dataclass(frozen=True)
class ProSSTSearchState:
    sequence: tuple[str, ...]
    mask_positions: tuple[int, ...]
    depth: int = 0
    score: float = 0.0
    log_likelihood: float = 0.0
    steps: tuple[ProSSTAction, ...] = ()


class ProSSTPUCTEvaluator:
    """Adapter from the generic PUCT search to ProSST masked-token decoding."""

    def __init__(self, generator, original_sequence: str, mask_positions):
        self.generator = generator
        self.original_sequence = original_sequence
        self.mask_positions = tuple(int(pos) for pos in mask_positions)
        self._actions_cache: dict[ProSSTSearchState, tuple[ActionPrior[ProSSTAction], ...]] = {}

    def is_terminal(self, state: ProSSTSearchState) -> bool:
        return state.depth >= len(state.mask_positions)

    def terminal_value(self, state: ProSSTSearchState) -> float:
        return float(state.score)

    def actions(self, state: ProSSTSearchState):
        cached = self._actions_cache.get(state)
        if cached is not None:
            return cached
        if self.is_terminal(state):
            return ()
        pos = self.mask_positions[state.depth]
        original_aa = self.original_sequence[pos]
        logits = self._masked_position_logits(state.sequence, pos)
        actions = tuple(self._actions_from_logits(logits, original_aa))
        self._actions_cache[state] = actions
        return actions

    def transition(self, state: ProSSTSearchState, action: ProSSTAction, immediate_value: float) -> ProSSTSearchState:
        pos = self.mask_positions[state.depth]
        sequence = list(state.sequence)
        sequence[pos] = action.aa
        return ProSSTSearchState(
            sequence=tuple(sequence),
            mask_positions=state.mask_positions,
            depth=state.depth + 1,
            score=state.score + float(action.log_delta),
            log_likelihood=state.log_likelihood + float(action.full_vocab_logp),
            steps=(*state.steps, action),
        )

    def rollout_value(self, state: ProSSTSearchState) -> float:
        return float(self.greedy_complete(state).score)

    def greedy_complete(self, state: ProSSTSearchState) -> ProSSTSearchState:
        current = state
        while not self.is_terminal(current):
            actions = list(self.actions(current))
            if not actions:
                break
            best = max(actions, key=lambda item: item.immediate_value)
            current = self.transition(current, best.action, best.immediate_value)
        return current

    def _masked_position_logits(self, sequence: tuple[str, ...], pos: int):
        text = "".join(sequence)
        input_ids, attention_mask = self.generator._tokenize_batch([text])
        input_ids = input_ids.clone()
        input_ids[0, pos + 1] = self.generator.tokenizer.mask_token_id
        logits = self.generator.logits_for_input_ids(input_ids, attention_mask)
        return logits[0, pos]

    def _actions_from_logits(self, logits, original_aa: str):
        generator = self.generator
        dist_ids = generator._distribution_ids(original_aa)
        dist_logits = logits[dist_ids].clone()
        if generator.args.smc_vocab == "full" and generator.args.non_cluster_logit_penalty > 0:
            cluster = set(generator.alphabet[original_aa])
            penalty = torch.tensor(
                [
                    0.0 if generator.id_to_aa[int(token.item())] in cluster else float(generator.args.non_cluster_logit_penalty)
                    for token in dist_ids
                ],
                dtype=dist_logits.dtype,
                device=dist_logits.device,
            )
            dist_logits = dist_logits - penalty
        dist_log_probs = F.log_softmax(dist_logits, dim=0)
        dist_probs = dist_log_probs.exp()
        original_id = torch.tensor(generator.aa_to_id[original_aa], dtype=torch.long, device=generator.device)
        original_idx = torch.nonzero(dist_ids == original_id, as_tuple=False).flatten()
        if original_idx.numel() == 0:
            raise ValueError(f"Original AA {original_aa} absent from sampling distribution")
        logp_original = dist_log_probs[original_idx[:1]].flatten()[0]
        full_log_probs = F.log_softmax(logits[generator.full_ids], dim=0)
        action_priors = []
        for idx, token_id in enumerate(dist_ids):
            sampled_id = int(token_id.item())
            aa = generator.id_to_aa[sampled_id]
            logp_sampled = dist_log_probs[idx]
            action = ProSSTAction(
                aa=aa,
                log_delta=float((logp_sampled - logp_original).detach().cpu()),
                full_vocab_logp=float(full_log_probs[AA20.index(aa)].detach().cpu()),
                logp_sampled=float(logp_sampled.detach().cpu()),
                logp_original=float(logp_original.detach().cpu()),
                prior=float(dist_probs[idx].detach().cpu()),
            )
            action_priors.append(ActionPrior(action=action, prior=action.prior, immediate_value=action.log_delta))
        return action_priors


def decode_with_puct(generator, covered_start: str, mask_positions, simulations: int, c_puct: float):
    evaluator = ProSSTPUCTEvaluator(generator, covered_start, mask_positions)
    initial_state = ProSSTSearchState(
        sequence=tuple(covered_start),
        mask_positions=tuple(int(pos) for pos in mask_positions),
    )
    search = PUCTSearch(evaluator, PUCTConfig(simulations=simulations, c_puct=c_puct))
    results = search.run(initial_state)
    if not results:
        final_state = evaluator.greedy_complete(initial_state)
        results = [type("FallbackResult", (), {"state": final_state, "value": final_state.score, "visits": 0})()]

    terminal_scores = np.asarray([result.value for result in results], dtype=float)
    best = results[0].state
    summary = {
        "mcts_simulations": int(simulations),
        "mcts_c_puct": float(c_puct),
        "terminal_count": int(len(results)),
        "terminal_positive_fraction": float(np.mean(terminal_scores > 0.0)) if len(terminal_scores) else 0.0,
        "terminal_score_mean": float(np.mean(terminal_scores)) if len(terminal_scores) else 0.0,
        "terminal_score_max": float(np.max(terminal_scores)) if len(terminal_scores) else 0.0,
        "terminal_score_min": float(np.min(terminal_scores)) if len(terminal_scores) else 0.0,
        "root_visits": int(search.root.visits) if search.root is not None else 0,
    }
    return best, summary

