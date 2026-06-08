from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from prospero.inference import ProteinSampler


def _with_mutation(sequence: str, position: int, residue: str) -> str:
    if sequence[position] == residue:
        return sequence
    chars = list(sequence)
    chars[position] = residue
    return "".join(chars)


def _split_budget(total: int) -> tuple[int, int, int]:
    n_1aa = int(np.floor(total * 0.50))
    n_2aa = int(np.floor(total * 0.25))
    n_3aa = total - n_1aa - n_2aa
    return n_1aa, n_2aa, n_3aa


def _hamming_distance(lhs: str, rhs: str) -> int:
    return sum(int(a != b) for a, b in zip(lhs, rhs))


@dataclass(frozen=True)
class _Candidate:
    sequence: str
    mutated_positions: frozenset[int]


class OADMSearchMethod:
    def __init__(
        self,
        *,
        oadm_model,
        oadm_tokenizer,
        alphabet,
        batch_size: int,
        resampling_steps: int,
        min_corruptions: int,
        max_corruptions: int,
        kappa_scan: float,
        n_checks_multiplier: int,
        kappa_guidance: float,
    ) -> None:
        self.oadm_model = oadm_model
        self.oadm_tokenizer = oadm_tokenizer
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.resampling_steps = resampling_steps
        self.min_corruptions = min_corruptions
        self.max_corruptions = max_corruptions
        self.kappa_scan = kappa_scan
        self.n_checks_multiplier = n_checks_multiplier
        self.kappa_guidance = kappa_guidance
        self._last_debug_summary: dict[str, object] = {}

    def get_last_debug_summary(self) -> dict[str, object]:
        return dict(self._last_debug_summary)

    def generate_candidates(
        self,
        *,
        proxy,
        starting_sequence: str,
        n_queries: int,
        ref_sequences: Iterable[str],
    ) -> list[str]:
        sampler = ProteinSampler(self.oadm_model, self.oadm_tokenizer, self.alphabet)
        sequences: list[str] = []
        seen = set(ref_sequences)
        rounds = 0

        while len(sequences) < n_queries:
            rounds += 1
            sampler.generate_raa_from_alanine_scan(
                proxy,
                starting_sequence,
                self.batch_size,
                self.resampling_steps,
                self.min_corruptions,
                self.max_corruptions,
                self.kappa_scan,
                self.n_checks_multiplier,
                self.kappa_guidance,
            )
            top_sequences = sampler.get_top_sequences(n_queries, seen)
            if not top_sequences:
                break
            for sequence in top_sequences:
                if sequence in seen:
                    continue
                seen.add(sequence)
                sequences.append(sequence)
                if len(sequences) >= n_queries:
                    break
        self._last_debug_summary = {
            "method": "oadm_smc",
            "requested": int(n_queries),
            "returned": int(len(sequences[:n_queries])),
            "rounds": int(rounds),
        }
        return sequences[:n_queries]


class LinearGreedySearchMethod:
    def __init__(
        self,
        *,
        allow_non_positive_fill: bool = True,
        substitution_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.allow_non_positive_fill = allow_non_positive_fill
        self.substitution_map = substitution_map or {}
        self._last_debug_summary: dict[str, object] = {}

    def get_last_debug_summary(self) -> dict[str, object]:
        return dict(self._last_debug_summary)

    def _extract_additive_weights(self, proxy, seq_length: int) -> tuple[np.ndarray, str]:
        models = getattr(proxy, "models", None)
        if not models:
            raise ValueError(
                "linear_greedy search requires an ensemble-like proxy with `models`."
            )

        all_weights = []
        alphabet = None
        for model in models:
            if not getattr(model, "_is_fitted", False):
                raise ValueError(
                    "linear_greedy search requires fitted linear/ridge surrogate models."
                )
            if not hasattr(model, "regressor") or not hasattr(model, "scaler"):
                raise ValueError(
                    "linear_greedy search requires one-hot sklearn-like models "
                    "(missing `regressor`/`scaler`)."
                )
            if not hasattr(model.regressor, "coef_"):
                raise ValueError(
                    "linear_greedy search requires linear/ridge surrogates exposing `coef_`."
                )

            model_alphabet = getattr(model, "alphabet", None)
            if model_alphabet is None:
                raise ValueError("Surrogate model is missing `alphabet`.")
            if alphabet is None:
                alphabet = model_alphabet
            elif alphabet != model_alphabet:
                raise ValueError("All ensemble models must use the same alphabet.")

            coef = np.asarray(model.regressor.coef_, dtype=np.float64).reshape(-1)
            scale = np.asarray(model.scaler.scale_, dtype=np.float64).reshape(-1)
            if coef.shape[0] != scale.shape[0]:
                raise ValueError("Regressor/scaler feature dimensions do not match.")
            safe_scale = np.where(scale == 0.0, 1.0, scale)
            effective_coef = coef / safe_scale

            expected_features = seq_length * len(alphabet)
            if effective_coef.shape[0] != expected_features:
                raise ValueError(
                    "linear_greedy currently supports one-hot-only surrogates. "
                    f"Expected {expected_features} features, got {effective_coef.shape[0]}."
                )
            all_weights.append(effective_coef.reshape(seq_length, len(alphabet)))

        return np.mean(np.stack(all_weights, axis=0), axis=0), alphabet

    def _best_mutation_per_position(
        self,
        *,
        sequence: str,
        weights: np.ndarray,
        alphabet: str,
        blocked_positions: set[int],
        positive_only: bool,
    ) -> list[tuple[float, int, str]]:
        aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}
        best: list[tuple[float, int, str]] = []
        for position, current_aa in enumerate(sequence):
            if position in blocked_positions:
                continue
            current_idx = aa_to_idx[current_aa]
            current_weight = weights[position, current_idx]
            best_gain = None
            best_aa = None
            allowed = self.substitution_map.get(current_aa, list(alphabet))
            allowed_set = set(allowed)
            for aa_idx, candidate_aa in enumerate(alphabet):
                if candidate_aa == current_aa:
                    continue
                if candidate_aa not in allowed_set:
                    continue
                gain = float(weights[position, aa_idx] - current_weight)
                if positive_only and gain <= 0.0:
                    continue
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_aa = candidate_aa
            if best_gain is not None and best_aa is not None:
                best.append((best_gain, position, best_aa))
        best.sort(key=lambda item: item[0], reverse=True)
        return best

    def _all_ranked_mutations(
        self,
        *,
        sequence: str,
        weights: np.ndarray,
        alphabet: str,
        blocked_positions: set[int],
        positive_only: bool,
    ) -> list[tuple[float, int, str]]:
        aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}
        ranked: list[tuple[float, int, str]] = []
        for position, current_aa in enumerate(sequence):
            if position in blocked_positions:
                continue
            current_idx = aa_to_idx[current_aa]
            current_weight = weights[position, current_idx]
            allowed = self.substitution_map.get(current_aa, list(alphabet))
            allowed_set = set(allowed)
            for aa_idx, candidate_aa in enumerate(alphabet):
                if candidate_aa == current_aa:
                    continue
                if candidate_aa not in allowed_set:
                    continue
                gain = float(weights[position, aa_idx] - current_weight)
                if positive_only and gain <= 0.0:
                    continue
                ranked.append((gain, position, candidate_aa))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _build_1aa_candidates(
        self,
        *,
        starting_sequence: str,
        weights: np.ndarray,
        alphabet: str,
        n_1aa: int,
        positive_only: bool,
    ) -> tuple[list[_Candidate], list[float]]:
        ranked = self._best_mutation_per_position(
            sequence=starting_sequence,
            weights=weights,
            alphabet=alphabet,
            blocked_positions=set(),
            positive_only=positive_only,
        )
        if not ranked:
            return []

        selected: list[tuple[float, int, str]] = []
        seen_positions: set[int] = set()
        for mutation in ranked:
            _, position, _ = mutation
            if position in seen_positions:
                continue
            selected.append(mutation)
            seen_positions.add(position)
            if len(selected) >= n_1aa:
                break
        if len(selected) < n_1aa:
            all_ranked = self._all_ranked_mutations(
                sequence=starting_sequence,
                weights=weights,
                alphabet=alphabet,
                blocked_positions=set(),
                positive_only=positive_only,
            )
            for mutation in all_ranked:
                if mutation in selected:
                    continue
                selected.append(mutation)
                if len(selected) >= n_1aa:
                    break

        candidates = []
        gains = []
        for _, position, aa in selected[:n_1aa]:
            sequence = _with_mutation(starting_sequence, position, aa)
            candidates.append(
                _Candidate(sequence=sequence, mutated_positions=frozenset({position}))
            )
            aa_to_idx = {token: idx for idx, token in enumerate(alphabet)}
            old_idx = aa_to_idx[starting_sequence[position]]
            new_idx = aa_to_idx[aa]
            gains.append(float(weights[position, new_idx] - weights[position, old_idx]))
        return candidates, gains

    def _enumerate_fallback_sequences(
        self,
        *,
        starting_sequence: str,
        weights: np.ndarray,
        alphabet: str,
    ) -> list[str]:
        singles = self._all_ranked_mutations(
            sequence=starting_sequence,
            weights=weights,
            alphabet=alphabet,
            blocked_positions=set(),
            positive_only=False,
        )
        top_per_position = self._best_mutation_per_position(
            sequence=starting_sequence,
            weights=weights,
            alphabet=alphabet,
            blocked_positions=set(),
            positive_only=False,
        )

        extras: list[str] = []
        # 1-AA fallback (all ranked singles)
        for _, position, aa in singles:
            extras.append(_with_mutation(starting_sequence, position, aa))

        # 2-AA fallback from best per-position mutations.
        top_for_pairs = top_per_position[:24]
        for i in range(len(top_for_pairs)):
            _, pos_i, aa_i = top_for_pairs[i]
            seq_i = _with_mutation(starting_sequence, pos_i, aa_i)
            for j in range(i + 1, len(top_for_pairs)):
                _, pos_j, aa_j = top_for_pairs[j]
                extras.append(_with_mutation(seq_i, pos_j, aa_j))

        # 3-AA fallback from the same ordered pool (kept bounded).
        top_for_triples = top_per_position[:16]
        for i in range(len(top_for_triples)):
            _, pos_i, aa_i = top_for_triples[i]
            seq_i = _with_mutation(starting_sequence, pos_i, aa_i)
            for j in range(i + 1, len(top_for_triples)):
                _, pos_j, aa_j = top_for_triples[j]
                seq_ij = _with_mutation(seq_i, pos_j, aa_j)
                for k in range(j + 1, len(top_for_triples)):
                    _, pos_k, aa_k = top_for_triples[k]
                    extras.append(_with_mutation(seq_ij, pos_k, aa_k))
        return extras

    def _expand_candidates(
        self,
        *,
        seeds: list[_Candidate],
        target_count: int,
        weights: np.ndarray,
        alphabet: str,
        positive_only: bool,
    ) -> tuple[list[_Candidate], list[float]]:
        if target_count <= 0 or not seeds:
            return []

        expanded: list[_Candidate] = []
        gains: list[float] = []
        seen_sequences = {seed.sequence for seed in seeds}
        seed_index = 0
        no_progress_rounds = 0
        max_no_progress_rounds = len(seeds) + 1

        while len(expanded) < target_count and no_progress_rounds < max_no_progress_rounds:
            seed = seeds[seed_index % len(seeds)]
            seed_index += 1
            ranked = self._best_mutation_per_position(
                sequence=seed.sequence,
                weights=weights,
                alphabet=alphabet,
                blocked_positions=set(seed.mutated_positions),
                positive_only=positive_only,
            )
            top_ranked = ranked[:3]
            before = len(expanded)
            for _, position, aa in top_ranked:
                new_sequence = _with_mutation(seed.sequence, position, aa)
                if new_sequence in seen_sequences:
                    continue
                seen_sequences.add(new_sequence)
                expanded.append(
                    _Candidate(
                        sequence=new_sequence,
                        mutated_positions=frozenset(set(seed.mutated_positions) | {position}),
                    )
                )
                gains.append(float(_))
                if len(expanded) >= target_count:
                    break
            if len(expanded) == before:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0

        return expanded[:target_count], gains[:target_count]

    def generate_candidates(
        self,
        *,
        proxy,
        starting_sequence: str,
        n_queries: int,
        ref_sequences: Iterable[str],
    ) -> list[str]:
        seq_length = len(starting_sequence)
        weights, alphabet = self._extract_additive_weights(proxy, seq_length=seq_length)

        n_1aa, n_2aa, n_3aa = _split_budget(n_queries)

        one_aa, one_aa_gains = self._build_1aa_candidates(
            starting_sequence=starting_sequence,
            weights=weights,
            alphabet=alphabet,
            n_1aa=n_1aa,
            positive_only=True,
        )
        two_aa, two_aa_gains = self._expand_candidates(
            seeds=one_aa,
            target_count=n_2aa,
            weights=weights,
            alphabet=alphabet,
            positive_only=True,
        )
        three_aa, three_aa_gains = self._expand_candidates(
            seeds=two_aa,
            target_count=n_3aa,
            weights=weights,
            alphabet=alphabet,
            positive_only=True,
        )

        if self.allow_non_positive_fill:
            if len(one_aa) < n_1aa:
                fill_one_aa, fill_one_aa_gains = self._build_1aa_candidates(
                        starting_sequence=starting_sequence,
                        weights=weights,
                        alphabet=alphabet,
                        n_1aa=n_1aa,
                        positive_only=False,
                    )
                one_aa.extend(fill_one_aa[len(one_aa) : n_1aa])
                one_aa_gains.extend(fill_one_aa_gains[len(one_aa_gains) : n_1aa])
            if len(two_aa) < n_2aa:
                fill_two_aa, fill_two_aa_gains = self._expand_candidates(
                        seeds=one_aa,
                        target_count=n_2aa,
                        weights=weights,
                        alphabet=alphabet,
                        positive_only=False,
                    )
                two_aa.extend(fill_two_aa[len(two_aa) : n_2aa])
                two_aa_gains.extend(fill_two_aa_gains[len(two_aa_gains) : n_2aa])
            if len(three_aa) < n_3aa:
                fill_three_aa, fill_three_aa_gains = self._expand_candidates(
                        seeds=two_aa,
                        target_count=n_3aa,
                        weights=weights,
                        alphabet=alphabet,
                        positive_only=False,
                    )
                three_aa.extend(fill_three_aa[len(three_aa) : n_3aa])
                three_aa_gains.extend(fill_three_aa_gains[len(three_aa_gains) : n_3aa])

        combined = one_aa + two_aa + three_aa
        ref = set(ref_sequences)
        unique_sequences = []
        seen = set()
        for candidate in combined:
            sequence = candidate.sequence
            if sequence in ref or sequence in seen:
                continue
            mutation_count = _hamming_distance(starting_sequence, sequence)
            if mutation_count not in {1, 2, 3}:
                continue
            unique_sequences.append(sequence)
            seen.add(sequence)

        fallback_added = 0
        if len(unique_sequences) < n_queries:
            for sequence in self._enumerate_fallback_sequences(
                starting_sequence=starting_sequence,
                weights=weights,
                alphabet=alphabet,
            ):
                if sequence in ref or sequence in seen:
                    continue
                mutation_count = _hamming_distance(starting_sequence, sequence)
                if mutation_count not in {1, 2, 3}:
                    continue
                unique_sequences.append(sequence)
                seen.add(sequence)
                fallback_added += 1
                if len(unique_sequences) >= n_queries:
                    break

        if not unique_sequences:
            return []

        with torch.no_grad():
            scores = proxy.get_scores(unique_sequences).detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        ranked_sequences = [unique_sequences[idx] for idx in order]
        returned = ranked_sequences[:n_queries]
        returned_mut_counts = [
            _hamming_distance(starting_sequence, sequence) for sequence in returned
        ]
        self._last_debug_summary = {
            "method": "linear_greedy",
            "requested": int(n_queries),
            "split_target": {"1aa": int(n_1aa), "2aa": int(n_2aa), "3aa": int(n_3aa)},
            "generated_before_filter": {
                "1aa": int(len(one_aa)),
                "2aa": int(len(two_aa)),
                "3aa": int(len(three_aa)),
            },
            "gain_means": {
                "1aa": float(np.mean(one_aa_gains)) if one_aa_gains else None,
                "2aa": float(np.mean(two_aa_gains)) if two_aa_gains else None,
                "3aa": float(np.mean(three_aa_gains)) if three_aa_gains else None,
            },
            "returned": int(len(returned)),
            "returned_mutation_counts": {
                "1aa": int(sum(m == 1 for m in returned_mut_counts)),
                "2aa": int(sum(m == 2 for m in returned_mut_counts)),
                "3aa": int(sum(m == 3 for m in returned_mut_counts)),
            },
            "fallback_added": int(fallback_added),
        }
        return returned


def build_search_method(args, alphabet):
    if args.search_method == "linear_greedy":
        substitution_map = alphabet if isinstance(alphabet, dict) else None
        return LinearGreedySearchMethod(substitution_map=substitution_map)

    if args.search_method != "oadm_smc":
        raise ValueError(f"Unsupported search_method={args.search_method}")

    from evodiff.pretrained import OA_DM_38M  # type: ignore[reportMissingImports]

    model, _, tokenizer_oadm, _ = OA_DM_38M()
    model = model.cuda()
    return OADMSearchMethod(
        oadm_model=model,
        oadm_tokenizer=tokenizer_oadm,
        alphabet=alphabet,
        batch_size=args.batch_size,
        resampling_steps=args.resampling_steps,
        min_corruptions=args.min_corruptions,
        max_corruptions=args.max_corruptions,
        kappa_scan=args.kappa_scan,
        n_checks_multiplier=args.n_checks_multiplier,
        kappa_guidance=args.kappa_guidance,
    )
