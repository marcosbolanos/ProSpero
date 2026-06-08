import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)

class Sampler:
    def __init__(self, model, tokenizer, alphabet):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unrolled_scores = dict()
        self.alphabet = alphabet
        self.token_to_cluster = self.get_token_clusters()
        self.PAD = -1
        self.trace_writer = None
        self.trace_context = {}
        self.mask_position_reward_sum = {}
        self.mask_position_reward_count = {}
        self.mask_position_sample_count = {}

    def set_trace_writer(self, trace_writer):
        self.trace_writer = trace_writer

    def set_trace_context(self, **kwargs):
        self.trace_context = {k: v for k, v in kwargs.items() if v is not None}

    def trace_event(self, event, **payload):
        if self.trace_writer is None:
            return
        record = {"event": event, **self.trace_context, **payload}
        self.trace_writer.write(record)

    def update_mask_position_rewards(self, starting_sequence, sequences, scores):
        """Online mask prior: reward absolute positions that improved a query batch."""
        if not sequences:
            return
        baseline = float(np.mean(scores))
        for sequence, score in zip(sequences, scores):
            advantage = float(score) - baseline
            for pos, (before, after) in enumerate(zip(starting_sequence, sequence)):
                if before == after:
                    continue
                self.mask_position_reward_sum[pos] = (
                    self.mask_position_reward_sum.get(pos, 0.0) + advantage
                )
                self.mask_position_reward_count[pos] = (
                    self.mask_position_reward_count.get(pos, 0) + 1
                )

    def token_to_aa(self, token):
        token = int(token.item()) if hasattr(token, "item") else int(token)
        if 0 <= token < len(self.tokenizer.all_aas):
            return self.tokenizer.all_aas[token]
        return str(token)

    def get_token_clusters(self):
        aa_to_token = {i: c for c, i in enumerate(self.tokenizer.all_aas[:20])}
        tokens_alphabet = {aa_to_token[k]: self.tokenizer.tokenize([v]) for k, v in self.alphabet.items()}
        return tokens_alphabet

    def get_top_sequences(self, k_best, ref_sequences):
        ref_sequences = set(ref_sequences)
        sorted_seqs = sorted(self.unrolled_scores, key=lambda x: self.unrolled_scores[x], reverse=True)
        filtered_seqs = [seq for seq in sorted_seqs if seq not in ref_sequences]
        for rank, seq in enumerate(filtered_seqs[:k_best], start=1):
            self.trace_event(
                "candidate_selected_for_query",
                selected_rank=int(rank),
                sequence=seq,
                zero_shot_score=float(self.unrolled_scores[seq]),
            )
        self.unrolled_scores = dict()
        return filtered_seqs[:k_best]


    def is_resampling_step(self, step, resampling_steps):
        if isinstance(resampling_steps, list):
            return step in resampling_steps
        else:
            return not step % resampling_steps

    def sample_lin_indices(self, raw_scores):
        raw_scores = raw_scores.cpu()
        raw_scores -= raw_scores.min()
        raw_scores += 1e-8
        weights = raw_scores / raw_scores.sum()
        indices = torch.multinomial(weights, len(weights), replacement=True)
        return indices.to(self.device)

    def sample_n_corruptions_uniform(self, min_corruptions, max_corruptions):
        return np.random.randint(min_corruptions, max_corruptions + 1)    

    def sample_n_corruptions_clipped_normal(
        self,
        mean_corruptions,
        std_corruptions,
        min_corruptions,
        max_corruptions,
    ):
        if std_corruptions <= 0:
            n_corruptions = int(round(mean_corruptions))
        else:
            n_corruptions = int(round(np.random.normal(mean_corruptions, std_corruptions)))
        return int(np.clip(n_corruptions, min_corruptions, max_corruptions))


class ProteinSampler(Sampler):
    def __init__(self, model, tokenizer, alphabet):
        super().__init__(model, tokenizer, alphabet)
                
    def shotgun_alanine_scan(self, sequence, proxy, min_corruptions, max_corruptions, batch_size, n_checks_multiplier, k):
        """
        Targeted Masking (Algorithm 2)
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]

        batch_substituted = np.empty((batch_size * n_checks_multiplier, len(sequence)), dtype="U1")
        sampled_ids_list = list()
        for n in range(batch_size * n_checks_multiplier):
            n_corruptions = self.sample_n_corruptions_uniform(min_corruptions, max_corruptions) # line 2
            try:
                sampled_ids = np.random.choice(maskable_ids, n_corruptions, replace=False) # line 3
            except ValueError:
                sampled_ids = maskable_ids
            split_sequence = np.array(list(sequence))
            split_sequence[sampled_ids] = "A" # line 4
            batch_substituted[n] = split_sequence
            sampled_ids_list.append(sampled_ids)
            
        substituted_seqs = ["".join(s) for s in batch_substituted]
        mean, std = proxy.forward_with_uncertainty(substituted_seqs)
        ucb = (mean + k * std).detach().cpu().numpy() # line 5
        best_ids = np.argsort(ucb)[::-1][:batch_size]
        
        batch = np.tile(tokenized_seq, (batch_size, 1))
        ids = [sampled_ids_list[idx] for idx in best_ids]
        
        locs = list()
        og_tokens_to_clusters = list()
        for row, row_ids in zip(batch, ids):
            loc = np.array(sorted(row_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append({i: self.token_to_cluster[row[i]] for i in loc})
            row[loc] = self.tokenizer.mask_id # line 7
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = np.array([{k: torch.tensor(v, device=self.device) for k, v in item.items()} for item in og_tokens_to_clusters])

        return sample, locs, og_tokens_to_clusters

    def shotgun_alanine_scan_zero_shot_metadata(
        self,
        sequence,
        proxy,
        min_corruptions,
        max_corruptions,
        batch_size,
        n_checks_multiplier,
        k,
    ):
        """
        Targeted masking with original-token metadata needed for logP-delta
        sequence scoring.
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]

        batch_substituted = np.empty((batch_size * n_checks_multiplier, len(sequence)), dtype="U1")
        sampled_ids_list = []
        for n in range(batch_size * n_checks_multiplier):
            n_corruptions = self.sample_n_corruptions_uniform(min_corruptions, max_corruptions)
            try:
                sampled_ids = np.random.choice(maskable_ids, n_corruptions, replace=False)
            except ValueError:
                sampled_ids = maskable_ids
            split_sequence = np.array(list(sequence))
            split_sequence[sampled_ids] = "A"
            batch_substituted[n] = split_sequence
            sampled_ids_list.append(sampled_ids)

        substituted_seqs = ["".join(s) for s in batch_substituted]
        mean, std = proxy.forward_with_uncertainty(substituted_seqs)
        ucb = (mean + k * std).detach().cpu().numpy()
        best_ids = np.argsort(ucb)[::-1][:batch_size]

        batch = np.tile(tokenized_seq, (batch_size, 1))
        ids = [sampled_ids_list[idx] for idx in best_ids]

        locs = []
        og_tokens_to_clusters = []
        for row, row_ids in zip(batch, ids):
            loc = np.array(sorted(row_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append(
                {
                    i: {
                        "original_token": int(row[i]),
                        "cluster": self.token_to_cluster[row[i]],
                    }
                    for i in loc
                }
            )
            row[loc] = self.tokenizer.mask_id
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = self._zero_shot_metadata_to_device(og_tokens_to_clusters)
        self._trace_mask_selected(sequence, locs, "calibrated_random")

        return sample, locs, og_tokens_to_clusters

    def collect_targeted_mask_counts(
        self,
        sequence,
        proxy,
        min_corruptions,
        max_corruptions,
        batch_size,
        n_checks_multiplier,
        k,
        n_rounds,
    ):
        """
        Measure selected targeted-mask counts from Algorithm 2 without using the
        resulting masks for generation.
        """
        counts = []
        for _ in range(n_rounds):
            _, locs, _ = self.shotgun_alanine_scan(
                sequence,
                proxy,
                min_corruptions,
                max_corruptions,
                batch_size,
                n_checks_multiplier,
                k,
            )
            counts.extend((locs != self.PAD).sum(axis=1).tolist())
        return np.array(counts, dtype=int)

    def random_mask_scan_zero_shot(
        self,
        sequence,
        batch_size,
        mask_count_mean,
        mask_count_std,
        mask_count_min,
        mask_count_max,
    ):
        """
        Random masking for the zero-shot ablation. Counts are drawn from a
        clipped Normal matched to selected targeted-mask counts.
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]

        batch = np.tile(tokenized_seq, (batch_size, 1))
        locs = []
        og_tokens_to_clusters = []
        max_possible = len(maskable_ids)
        for row in batch:
            n_corruptions = self.sample_n_corruptions_clipped_normal(
                mask_count_mean,
                mask_count_std,
                mask_count_min,
                mask_count_max,
            )
            n_corruptions = min(n_corruptions, max_possible)
            sampled_ids = np.random.choice(maskable_ids, n_corruptions, replace=False)
            loc = np.array(sorted(sampled_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append(
                {
                    i: {
                        "original_token": int(row[i]),
                        "cluster": self.token_to_cluster[row[i]],
                    }
                    for i in loc
                }
            )
            row[loc] = self.tokenizer.mask_id
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = self._zero_shot_metadata_to_device(og_tokens_to_clusters)

        return sample, locs, og_tokens_to_clusters

    def fixed_mask_scan_zero_shot(
        self,
        sequence,
        batch_size,
        mask_budget,
        strategy="random",
        entropy_quantile=0.5,
        entropy_sigma=None,
        seed_grow_alpha=1.0,
        seed_grow_beta=1.0,
        coupling_tau=4.0,
    ):
        """
        Fixed-budget zero-shot masking strategies. No training data or surrogate
        is used; masks are selected only from the current sequence and EvoDiff.
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]
        if len(maskable_ids) == 0:
            raise ValueError("No maskable residues found in sequence.")
        mask_budget = int(min(max(1, mask_budget), len(maskable_ids)))

        base_scores = None
        entropy_metadata = None
        if strategy in {"middle_entropy", "seed_grow", "mixed_explore_exploit"}:
            entropies = self.compute_masked_position_entropies(sequence, maskable_ids)
            base_scores, entropy_metadata = self.middle_entropy_scores(
                entropies,
                quantile=entropy_quantile,
                sigma=entropy_sigma,
            )

        batch = np.tile(tokenized_seq, (batch_size, 1))
        locs = []
        og_tokens_to_clusters = []
        for row in batch:
            if strategy == "random":
                sampled_ids = np.random.choice(maskable_ids, mask_budget, replace=False)
            elif strategy == "middle_entropy":
                sampled_ids = np.random.choice(
                    maskable_ids,
                    mask_budget,
                    replace=False,
                    p=base_scores,
                )
            elif strategy == "seed_grow":
                sampled_ids = self.sample_seed_and_grow_mask(
                    maskable_ids,
                    base_scores,
                    mask_budget,
                    alpha=seed_grow_alpha,
                    beta=seed_grow_beta,
                    coupling_tau=coupling_tau,
                )
            elif strategy == "mixed_explore_exploit":
                sampled_ids = self.sample_mixed_explore_exploit_mask(
                    maskable_ids,
                    base_scores,
                    mask_budget,
                    alpha=seed_grow_alpha,
                    beta=seed_grow_beta,
                    coupling_tau=coupling_tau,
                )
            else:
                raise ValueError(f"Unsupported zero-shot mask strategy: {strategy}")

            loc = np.array(sorted(sampled_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append(
                {
                    i: {
                        "original_token": int(row[i]),
                        "cluster": self.token_to_cluster[row[i]],
                    }
                    for i in loc
                }
            )
            row[loc] = self.tokenizer.mask_id
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = self._zero_shot_metadata_to_device(og_tokens_to_clusters)
        self._trace_mask_selected(
            sequence,
            locs,
            strategy,
            entropy_metadata=entropy_metadata,
        )
        return sample, locs, og_tokens_to_clusters, entropy_metadata

    def _trace_mask_selected(self, starting_sequence, locs, strategy, entropy_metadata=None):
        if self.trace_writer is None:
            return
        for particle_idx, loc in enumerate(locs):
            positions = [int(pos) for pos in loc if pos != self.PAD]
            for pos in positions:
                self.mask_position_sample_count[pos] = (
                    self.mask_position_sample_count.get(pos, 0) + 1
                )
            self.trace_event(
                "mask_selected",
                particle=int(particle_idx),
                strategy=strategy,
                starting_sequence=starting_sequence,
                mask_positions=positions,
                mask_residues=[starting_sequence[pos] for pos in positions],
                mask_size=len(positions),
                entropy_metadata=entropy_metadata,
            )

    @torch.no_grad()
    def compute_masked_position_entropies(self, sequence, maskable_ids, chunk_size=64):
        tokenized_seq = self.tokenizer.tokenize([sequence])
        masked_rows = np.tile(tokenized_seq, (len(maskable_ids), 1))
        for row_idx, seq_idx in enumerate(maskable_ids):
            masked_rows[row_idx, seq_idx] = self.tokenizer.mask_id

        entropies = []
        for start in range(0, len(masked_rows), chunk_size):
            chunk = torch.tensor(masked_rows[start : start + chunk_size], device=self.device)
            timestep = torch.zeros(chunk.shape[0], dtype=torch.long, device=self.device)
            prediction = self.model(chunk, timestep)
            chunk_positions = maskable_ids[start : start + chunk_size]
            logits = prediction[torch.arange(chunk.shape[0], device=self.device), chunk_positions, :20]
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            entropies.append((-(probs * log_probs).sum(dim=1)).detach().cpu())
        return torch.cat(entropies).numpy()

    def middle_entropy_scores(self, entropies, quantile=0.5, sigma=None):
        h_star = float(np.quantile(entropies, quantile))
        if sigma is None:
            sigma = float(np.std(entropies))
            if sigma <= 1e-8:
                sigma = 1.0
        scores = np.exp(-((entropies - h_star) ** 2) / (2 * sigma**2))
        scores = scores + 1e-12
        scores = scores / scores.sum()
        return scores, {
            "entropy_quantile": float(quantile),
            "entropy_h_star": h_star,
            "entropy_sigma": float(sigma),
            "entropy_min": float(np.min(entropies)),
            "entropy_max": float(np.max(entropies)),
            "entropy_mean": float(np.mean(entropies)),
        }

    def sample_seed_and_grow_mask(
        self,
        maskable_ids,
        base_scores,
        mask_budget,
        alpha=1.0,
        beta=1.0,
        coupling_tau=4.0,
    ):
        selected = [int(np.random.choice(maskable_ids, p=base_scores))]
        maskable_ids = np.asarray(maskable_ids)
        base_scores = np.asarray(base_scores)
        while len(selected) < mask_budget:
            remaining_mask = ~np.isin(maskable_ids, selected)
            remaining_ids = maskable_ids[remaining_mask]
            remaining_base = base_scores[remaining_mask]
            distances = np.abs(remaining_ids[:, None] - np.asarray(selected)[None, :])
            coupling = np.exp(-distances / max(coupling_tau, 1e-6)).max(axis=1)
            scores = alpha * remaining_base + beta * coupling
            scores = np.maximum(scores, 1e-12)
            scores = scores / scores.sum()
            selected.append(int(np.random.choice(remaining_ids, p=scores)))
        return np.array(selected)

    def sample_mixed_explore_exploit_mask(
        self,
        maskable_ids,
        base_scores,
        mask_budget,
        alpha=1.0,
        beta=1.0,
        coupling_tau=4.0,
    ):
        """
        Mixed masking for low-data zero-shot FT:
        - roughly half exploit positions from online oracle reward memory,
        - one middle-entropy exploration position,
        - remaining positions random with an anti-collapse penalty.
        """
        maskable_ids = np.asarray(maskable_ids)
        base_scores = np.asarray(base_scores)
        selected = []

        def remaining():
            return ~np.isin(maskable_ids, selected)

        def choose(ids, scores):
            scores = np.asarray(scores, dtype=float)
            scores = np.maximum(scores, 1e-12)
            scores = scores / scores.sum()
            return int(np.random.choice(ids, p=scores))

        exploit_budget = max(1, mask_budget // 2)
        entropy_budget = 1 if mask_budget - exploit_budget > 0 else 0

        reward_scores = []
        for pos in maskable_ids:
            count = self.mask_position_reward_count.get(int(pos), 0)
            if count:
                reward_scores.append(self.mask_position_reward_sum.get(int(pos), 0.0) / count)
            else:
                reward_scores.append(0.0)
        reward_scores = np.asarray(reward_scores, dtype=float)
        positive = np.maximum(reward_scores, 0.0)

        # If there is no positive online signal yet, use seed-and-grow as the exploit fallback.
        if positive.sum() <= 1e-12:
            fallback = self.sample_seed_and_grow_mask(
                maskable_ids,
                base_scores,
                min(exploit_budget, mask_budget),
                alpha=alpha,
                beta=beta,
                coupling_tau=coupling_tau,
            )
            selected.extend([int(pos) for pos in fallback])
        else:
            for _ in range(min(exploit_budget, mask_budget)):
                rem = remaining()
                ids = maskable_ids[rem]
                if len(ids) == 0:
                    break
                scores = positive[rem]
                if scores.sum() <= 1e-12:
                    scores = base_scores[rem]
                selected.append(choose(ids, scores))

        for _ in range(entropy_budget):
            rem = remaining()
            ids = maskable_ids[rem]
            if len(ids) == 0 or len(selected) >= mask_budget:
                break
            selected.append(choose(ids, base_scores[rem]))

        while len(selected) < mask_budget:
            rem = remaining()
            ids = maskable_ids[rem]
            if len(ids) == 0:
                break
            counts = np.asarray(
                [self.mask_position_sample_count.get(int(pos), 0) for pos in ids],
                dtype=float,
            )
            anti_collapse = 1.0 / np.sqrt(1.0 + counts)
            selected.append(choose(ids, anti_collapse))

        return np.array(selected)

    def random_mask_scan(
        self,
        sequence,
        batch_size,
        mask_count_mean,
        mask_count_std,
        mask_count_min,
        mask_count_max,
    ):
        """
        Random masking with the original metadata format used by the standard
        surrogate-scored SMC path.
        """
        tokenized_seq = self.tokenizer.tokenize([sequence])
        maskable_tokens = np.array(list(self.token_to_cluster))
        maskable_ids = np.nonzero(np.isin(tokenized_seq, maskable_tokens))[0]

        batch = np.tile(tokenized_seq, (batch_size, 1))
        locs = []
        og_tokens_to_clusters = []
        max_possible = len(maskable_ids)
        for row in batch:
            n_corruptions = self.sample_n_corruptions_clipped_normal(
                mask_count_mean,
                mask_count_std,
                mask_count_min,
                mask_count_max,
            )
            n_corruptions = min(n_corruptions, max_possible)
            sampled_ids = np.random.choice(maskable_ids, n_corruptions, replace=False)
            loc = np.array(sorted(sampled_ids, key=lambda x: len(self.token_to_cluster[row[x]])))
            og_tokens_to_clusters.append({i: self.token_to_cluster[row[i]] for i in loc})
            row[loc] = self.tokenizer.mask_id
            locs.append(loc)

        max_loc_length = np.max([len(loc) for loc in locs])
        locs = np.array([np.pad(loc, (0, max_loc_length - len(loc)), constant_values=self.PAD) for loc in locs])
        sample = torch.tensor(batch, device=self.device)
        og_tokens_to_clusters = np.array([{k: torch.tensor(v, device=self.device) for k, v in item.items()} for item in og_tokens_to_clusters])

        return sample, locs, og_tokens_to_clusters

    def _zero_shot_metadata_to_device(self, og_tokens_to_clusters):
        return np.array(
            [
                {
                    k: {
                        "original_token": torch.tensor(v["original_token"], device=self.device),
                        "cluster": torch.tensor(v["cluster"], device=self.device),
                    }
                    for k, v in item.items()
                }
                for item in og_tokens_to_clusters
            ]
        )

    def _sample_and_score_logits(self, logits, original_token, distribution_tokens):
        dist_log_probs = F.log_softmax(logits[distribution_tokens], dim=0)
        dist_probs = dist_log_probs.exp()
        sampled_idx = torch.multinomial(dist_probs, num_samples=1)
        sampled_token = distribution_tokens[sampled_idx]

        original_idx = torch.nonzero(
            distribution_tokens == original_token,
            as_tuple=False,
        ).flatten()
        if original_idx.numel() == 0:
            raise ValueError("Original token is not present in the sampling distribution.")
        logp_sampled = dist_log_probs[sampled_idx]
        logp_original = dist_log_probs[original_idx[:1]]
        log_delta = logp_sampled - logp_original
        full_log_probs = F.log_softmax(logits[:20], dim=0)
        sampled_full_ll = full_log_probs[sampled_token]
        return (
            sampled_token,
            log_delta.flatten()[0],
            sampled_full_ll.flatten()[0],
            logp_sampled.flatten()[0],
            logp_original.flatten()[0],
        )

    @torch.no_grad
    def generate_zero_shot_from_random_masks(
        self,
        starting_sequence,
        batch_size,
        resampling_steps,
        mask_count_mean,
        mask_count_std,
        mask_count_min,
        mask_count_max,
    ):
        """
        Zero-shot SMC ablation: random masks, constrained unmasking for the main
        path, full-vocabulary rollouts, and cumulative log-probability deltas as
        the ranking score.
        """
        sample, locs, og_tokens_to_clusters = self.random_mask_scan_zero_shot(
            starting_sequence,
            batch_size,
            mask_count_mean,
            mask_count_std,
            mask_count_min,
            mask_count_max,
        )

        steps = len(locs[0])
        batch_ll = torch.zeros(batch_size, device=self.device)
        zero_shot_scores = torch.zeros(batch_size, device=self.device)
        n_predictions = torch.tensor((locs != self.PAD).sum(axis=1), device=self.device)
        for i in tqdm(range(steps)):
            steps_left = abs(i - steps)
            samples_left = np.nonzero(locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, locs[samples_left, i]]

            sampled_aas = []
            step_log_deltas = []
            step_ll = []
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], locs[samples_left, i]):
                mask_metadata = og_token_to_cluster[og_idx]
                original_token_tensor = mask_metadata["original_token"]
                cluster_tokens = mask_metadata["cluster"]
                sampled_token, log_delta, sampled_ll, _, _ = self._sample_and_score_logits(
                    logits,
                    original_token_tensor,
                    cluster_tokens,
                )
                sampled_aas.append(sampled_token)
                step_log_deltas.append(log_delta)
                step_ll.append(sampled_ll)
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, locs[samples_left, i]] = sampled_aas
            zero_shot_scores[samples_left] += torch.stack(step_log_deltas)
            batch_ll[samples_left] += torch.stack(step_ll)

            if self.is_resampling_step(steps_left, resampling_steps):
                unrolled_sample, unrolled_ll, unrolled_scores = self.unroll_zero_shot(
                    sample,
                    locs[:, i + 1 :],
                    og_tokens_to_clusters,
                    batch_ll,
                    zero_shot_scores,
                )
                inv_perplexity = 1 / torch.exp(-unrolled_ll / n_predictions)
                if steps_left < 10:
                    self.unrolled_scores |= dict(zip(unrolled_sample, unrolled_scores.detach().cpu().numpy()))

                ids = self.sample_lin_indices(unrolled_scores * inv_perplexity)
                sample = sample[ids]
                batch_ll = batch_ll[ids]
                zero_shot_scores = zero_shot_scores[ids]
                n_predictions = n_predictions[ids]
                ids = ids.cpu().numpy()
                locs = locs[ids]
                og_tokens_to_clusters = og_tokens_to_clusters[ids]

    @torch.no_grad
    def generate_zero_shot_from_fixed_masks(
        self,
        starting_sequence,
        batch_size,
        resampling_steps,
        mask_budget,
        strategy,
        entropy_quantile=0.5,
        entropy_sigma=None,
        seed_grow_alpha=1.0,
        seed_grow_beta=1.0,
        coupling_tau=4.0,
        smc_vocab="cluster",
        generation_mode="smc_rollout",
    ):
        sample, locs, og_tokens_to_clusters, _ = self.fixed_mask_scan_zero_shot(
            starting_sequence,
            batch_size,
            mask_budget,
            strategy=strategy,
            entropy_quantile=entropy_quantile,
            entropy_sigma=entropy_sigma,
            seed_grow_alpha=seed_grow_alpha,
            seed_grow_beta=seed_grow_beta,
            coupling_tau=coupling_tau,
        )
        self._generate_zero_shot_from_masked_sample(
            sample,
            locs,
            og_tokens_to_clusters,
            batch_size,
            resampling_steps,
            smc_vocab=smc_vocab,
            generation_mode=generation_mode,
        )

    @torch.no_grad
    def generate_zero_shot_from_targeted_scan(
        self,
        guide,
        starting_sequence,
        batch_size,
        resampling_steps,
        min_corruptions,
        max_corruptions,
        kappa_scan,
        n_checks_multiplier,
    ):
        """
        Targeted masking from the surrogate, then zero-shot-style logP-delta
        sequence scoring for SMC and final ranking.
        """
        sample, locs, og_tokens_to_clusters = self.shotgun_alanine_scan_zero_shot_metadata(
            starting_sequence,
            guide,
            min_corruptions,
            max_corruptions,
            batch_size,
            n_checks_multiplier,
            kappa_scan,
        )
        self._generate_zero_shot_from_masked_sample(
            sample,
            locs,
            og_tokens_to_clusters,
            batch_size,
            resampling_steps,
        )

    def _generate_zero_shot_from_masked_sample(
        self,
        sample,
        locs,
        og_tokens_to_clusters,
        batch_size,
        resampling_steps,
        smc_vocab="cluster",
        generation_mode="smc_rollout",
    ):
        if smc_vocab not in {"cluster", "full"}:
            raise ValueError(f"Unsupported SMC vocabulary: {smc_vocab}")
        if generation_mode not in {"smc_rollout", "no_rollout_sequential"}:
            raise ValueError(f"Unsupported zero-shot generation mode: {generation_mode}")
        full_tokens = torch.arange(20, device=self.device)
        steps = len(locs[0])
        batch_ll = torch.zeros(batch_size, device=self.device)
        zero_shot_scores = torch.zeros(batch_size, device=self.device)
        n_predictions = torch.tensor((locs != self.PAD).sum(axis=1), device=self.device)
        for i in tqdm(range(steps)):
            steps_left = abs(i - steps)
            samples_left = np.nonzero(locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, locs[samples_left, i]]

            sampled_aas = []
            step_log_deltas = []
            step_ll = []
            step_logp_sampled = []
            step_logp_original = []
            step_sampled_tokens = []
            step_original_tokens = []
            step_trace_payloads = []
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], locs[samples_left, i]):
                mask_metadata = og_token_to_cluster[og_idx]
                distribution_tokens = (
                    full_tokens if smc_vocab == "full" else mask_metadata["cluster"]
                )
                sampled_token, log_delta, sampled_ll, logp_sampled, logp_original = self._sample_and_score_logits(
                    logits,
                    mask_metadata["original_token"],
                    distribution_tokens,
                )
                sampled_aas.append(sampled_token)
                step_log_deltas.append(log_delta)
                step_ll.append(sampled_ll)
                step_logp_sampled.append(logp_sampled)
                step_logp_original.append(logp_original)
                step_sampled_tokens.append(sampled_token)
                step_original_tokens.append(mask_metadata["original_token"])
                step_trace_payloads.append(
                    {
                        "particle": int(samples_left[len(sampled_aas) - 1]),
                        "step": int(i + 1),
                        "position": int(og_idx),
                        "distribution": "full_20aa" if smc_vocab == "full" else "constrained_cluster",
                    }
                )
            self._trace_step_payloads(
                "smc_step",
                step_trace_payloads,
                step_log_deltas,
                step_ll,
                step_logp_sampled,
                step_logp_original,
                step_sampled_tokens,
                step_original_tokens,
            )
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, locs[samples_left, i]] = sampled_aas
            zero_shot_scores[samples_left] += torch.stack(step_log_deltas)
            batch_ll[samples_left] += torch.stack(step_ll)

            if generation_mode == "smc_rollout" and self.is_resampling_step(steps_left, resampling_steps):
                unrolled_sample, unrolled_ll, unrolled_scores = self.unroll_zero_shot(
                    sample,
                    locs[:, i + 1 :],
                    og_tokens_to_clusters,
                    batch_ll,
                    zero_shot_scores,
                )
                inv_perplexity = 1 / torch.exp(-unrolled_ll / n_predictions)
                self._trace_candidates(
                    unrolled_sample,
                    unrolled_scores,
                    unrolled_ll,
                    inv_perplexity,
                    stage="pre_resample",
                    smc_step=i + 1,
                )
                if steps_left < 10:
                    self.unrolled_scores |= dict(zip(unrolled_sample, unrolled_scores.detach().cpu().numpy()))

                raw_resampling_scores = unrolled_scores * inv_perplexity
                ids = self.sample_lin_indices(raw_resampling_scores)
                self._trace_resample(ids, raw_resampling_scores, smc_step=i + 1)
                sample = sample[ids]
                batch_ll = batch_ll[ids]
                zero_shot_scores = zero_shot_scores[ids]
                n_predictions = n_predictions[ids]
                ids = ids.cpu().numpy()
                locs = locs[ids]
                og_tokens_to_clusters = og_tokens_to_clusters[ids]

        if generation_mode == "no_rollout_sequential":
            sequences = [self.tokenizer.untokenize(seq) for seq in sample]
            inv_perplexity = 1 / torch.exp(-batch_ll / n_predictions)
            self._trace_candidates(
                sequences,
                zero_shot_scores,
                batch_ll,
                inv_perplexity,
                stage="terminal_no_rollout",
                smc_step=steps,
            )
            self.unrolled_scores |= dict(zip(sequences, zero_shot_scores.detach().cpu().numpy()))

    def _trace_candidates(self, sequences, scores, ll, inv_perplexity, stage, smc_step):
        if self.trace_writer is None:
            return
        scores = scores.detach().cpu().numpy()
        ll = ll.detach().cpu().numpy()
        inv_perplexity = inv_perplexity.detach().cpu().numpy()
        for candidate_idx, (sequence, score, log_likelihood, inv_ppl) in enumerate(
            zip(sequences, scores, ll, inv_perplexity)
        ):
            self.trace_event(
                "candidate",
                stage=stage,
                smc_step=int(smc_step),
                candidate=int(candidate_idx),
                sequence=sequence,
                zero_shot_score=float(score),
                log_likelihood=float(log_likelihood),
                inv_perplexity=float(inv_ppl),
            )

    def _trace_resample(self, parent_ids, raw_scores, smc_step):
        if self.trace_writer is None:
            return
        parent_ids_cpu = parent_ids.detach().cpu().numpy()
        raw_scores_cpu = raw_scores.detach().cpu()
        shifted = raw_scores_cpu - raw_scores_cpu.min() + 1e-8
        weights = shifted / shifted.sum()
        weights_np = weights.numpy()
        for child_idx, parent_idx in enumerate(parent_ids_cpu):
            self.trace_event(
                "resample",
                smc_step=int(smc_step),
                child=int(child_idx),
                parent=int(parent_idx),
                parent_weight=float(weights_np[parent_idx]),
            )

    def _trace_step_payloads(
        self,
        event,
        payloads,
        log_deltas,
        sampled_ll,
        logp_sampled,
        logp_original,
        sampled_tokens,
        original_tokens,
    ):
        if self.trace_writer is None or not payloads:
            return
        log_deltas = torch.stack(log_deltas).detach().cpu().numpy()
        sampled_ll = torch.stack(sampled_ll).detach().cpu().numpy()
        logp_sampled = torch.stack(logp_sampled).detach().cpu().numpy()
        logp_original = torch.stack(logp_original).detach().cpu().numpy()
        sampled_tokens = torch.cat(sampled_tokens).detach().cpu().numpy()
        original_tokens = torch.stack(original_tokens).detach().cpu().numpy()
        for idx, payload in enumerate(payloads):
            self.trace_event(
                event,
                **payload,
                sampled=self.token_to_aa(sampled_tokens[idx]),
                original=self.token_to_aa(original_tokens[idx]),
                logp_sampled=float(logp_sampled[idx]),
                logp_original=float(logp_original[idx]),
                log_delta=float(log_deltas[idx]),
                full_vocab_logp_sampled=float(sampled_ll[idx]),
            )

    @torch.no_grad
    def unroll_zero_shot(
        self,
        sample,
        remaining_locs,
        og_tokens_to_clusters,
        batch_ll,
        zero_shot_scores,
    ):
        """
        Zero-shot rollout: remaining masked residues are sampled from the full
        20-AA vocabulary and scored by logP(sampled)-logP(original).
        """
        batch_ll = deepcopy(batch_ll)
        zero_shot_scores = deepcopy(zero_shot_scores)
        sample = deepcopy(sample)
        full_tokens = torch.arange(20, device=self.device)
        for i in range(len(remaining_locs[0])):
            samples_left = np.nonzero(remaining_locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, remaining_locs[samples_left, i]]

            sampled_aas = []
            step_log_deltas = []
            step_ll = []
            step_logp_sampled = []
            step_logp_original = []
            step_sampled_tokens = []
            step_original_tokens = []
            step_trace_payloads = []
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], remaining_locs[samples_left, i]):
                mask_metadata = og_token_to_cluster[og_idx]
                original_token_tensor = mask_metadata["original_token"]
                sampled_token, log_delta, sampled_ll, logp_sampled, logp_original = self._sample_and_score_logits(
                    logits,
                    original_token_tensor,
                    full_tokens,
                )
                sampled_aas.append(sampled_token)
                step_log_deltas.append(log_delta)
                step_ll.append(sampled_ll)
                step_logp_sampled.append(logp_sampled)
                step_logp_original.append(logp_original)
                step_sampled_tokens.append(sampled_token)
                step_original_tokens.append(original_token_tensor)
                step_trace_payloads.append(
                    {
                        "particle": int(samples_left[len(sampled_aas) - 1]),
                        "rollout_step": int(i + 1),
                        "position": int(og_idx),
                        "distribution": "full_20aa",
                    }
                )
            self._trace_step_payloads(
                "rollout_step",
                step_trace_payloads,
                step_log_deltas,
                step_ll,
                step_logp_sampled,
                step_logp_original,
                step_sampled_tokens,
                step_original_tokens,
            )
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, remaining_locs[samples_left, i]] = sampled_aas
            zero_shot_scores[samples_left] += torch.stack(step_log_deltas)
            batch_ll[samples_left] += torch.stack(step_ll)

        untokenized = [self.tokenizer.untokenize(s) for s in sample]
        return untokenized, batch_ll, zero_shot_scores
  
    @torch.no_grad
    def generate_raa_from_alanine_scan(
        self, guide, starting_sequence, batch_size, resampling_steps, min_corruptions, max_corruptions, kappa_scan, n_checks_multiplier, kappa_guidance
    ):
        """
        Biologically-constrained SMC (Algorithm 3)
        """
        sample, locs, og_tokens_to_clusters = self.shotgun_alanine_scan(
            starting_sequence, guide, min_corruptions, max_corruptions, batch_size, n_checks_multiplier, kappa_scan
        )
        
        steps = len(locs[0])
        batch_ll = torch.zeros(batch_size, device=self.device)
        n_predictions = torch.tensor((locs != self.PAD).sum(axis=1), device=self.device)
        for i in tqdm(range(steps)):
            steps_left = abs(i - steps)
            samples_left = np.nonzero(locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, locs[samples_left, i]]

            sampled_aas = list()
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], locs[samples_left, i]):
                probs = torch.nn.functional.softmax(logits[og_token_to_cluster[og_idx]], dim=0) # line 10
                aa_idx = torch.multinomial(probs, num_samples=1)
                sampled_aas.append(og_token_to_cluster[og_idx][aa_idx])
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, locs[samples_left, i]] = sampled_aas

            log_probs = F.log_softmax(p[:, :20], dim=1)
            ll = log_probs[torch.arange(p.shape[0]), sampled_aas] # line 11
            batch_ll[samples_left] += ll
            
            if self.is_resampling_step(steps_left, resampling_steps):
                unrolled_sample, unrolled_ll = self.unroll_from_alanine_scan(sample, locs[:, i+1:], og_tokens_to_clusters, batch_ll) # line 12
                inv_perplexity = 1 / torch.exp(-unrolled_ll / n_predictions) # line 14
                scores = guide.get_ucb(unrolled_sample, kappa_guidance) # line 13
                if steps_left < 10:
                    self.unrolled_scores |= dict(zip(unrolled_sample, scores))
            
                # resample
                ids = self.sample_lin_indices(scores * inv_perplexity)
                sample = sample[ids]
                batch_ll = batch_ll[ids]
                n_predictions = n_predictions[ids]
                ids = ids.cpu().numpy()
                locs = locs[ids]
                og_tokens_to_clusters = og_tokens_to_clusters[ids]

    @torch.no_grad
    def generate_raa_from_random_masks(
        self,
        guide,
        starting_sequence,
        batch_size,
        resampling_steps,
        mask_count_mean,
        mask_count_std,
        mask_count_min,
        mask_count_max,
        kappa_guidance,
    ):
        """
        Standard surrogate-scored SMC, but replacing targeted masking with
        calibrated random masking.
        """
        sample, locs, og_tokens_to_clusters = self.random_mask_scan(
            starting_sequence,
            batch_size,
            mask_count_mean,
            mask_count_std,
            mask_count_min,
            mask_count_max,
        )

        steps = len(locs[0])
        batch_ll = torch.zeros(batch_size, device=self.device)
        n_predictions = torch.tensor((locs != self.PAD).sum(axis=1), device=self.device)
        for i in tqdm(range(steps)):
            steps_left = abs(i - steps)
            samples_left = np.nonzero(locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, locs[samples_left, i]]

            sampled_aas = []
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], locs[samples_left, i]):
                probs = torch.nn.functional.softmax(logits[og_token_to_cluster[og_idx]], dim=0)
                aa_idx = torch.multinomial(probs, num_samples=1)
                sampled_aas.append(og_token_to_cluster[og_idx][aa_idx])
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, locs[samples_left, i]] = sampled_aas

            log_probs = F.log_softmax(p[:, :20], dim=1)
            ll = log_probs[torch.arange(p.shape[0]), sampled_aas]
            batch_ll[samples_left] += ll

            if self.is_resampling_step(steps_left, resampling_steps):
                unrolled_sample, unrolled_ll = self.unroll_from_alanine_scan(sample, locs[:, i + 1 :], og_tokens_to_clusters, batch_ll)
                inv_perplexity = 1 / torch.exp(-unrolled_ll / n_predictions)
                scores = guide.get_ucb(unrolled_sample, kappa_guidance)
                if steps_left < 10:
                    self.unrolled_scores |= dict(zip(unrolled_sample, scores))

                ids = self.sample_lin_indices(scores * inv_perplexity)
                sample = sample[ids]
                batch_ll = batch_ll[ids]
                n_predictions = n_predictions[ids]
                ids = ids.cpu().numpy()
                locs = locs[ids]
                og_tokens_to_clusters = og_tokens_to_clusters[ids]
    

    @torch.no_grad
    def unroll_from_alanine_scan(self, sample, remaining_locs, og_tokens_to_clusters, batch_ll):
        """
        Rollout (Algorithm 4)
        """
        batch_ll = deepcopy(batch_ll)
        sample = deepcopy(sample)
        for i in range(len(remaining_locs[0])):
            samples_left = np.nonzero(remaining_locs[:, i] != self.PAD)[0]
            if not len(samples_left):
                break
            timestep = torch.tensor([0] * len(samples_left)).to(self.device)
            prediction = self.model(sample, timestep)
            p = prediction[samples_left, remaining_locs[samples_left, i]]

            sampled_aas = list()
            for logits, og_token_to_cluster, og_idx in zip(p, og_tokens_to_clusters[samples_left], remaining_locs[samples_left, i]):
                probs = torch.nn.functional.softmax(logits[og_token_to_cluster[og_idx]], dim=0) # line 5
                aa_idx = torch.multinomial(probs, num_samples=1)
                sampled_aas.append(og_token_to_cluster[og_idx][aa_idx])
            sampled_aas = torch.cat(sampled_aas)
            sample[samples_left, remaining_locs[samples_left, i]] = sampled_aas

            log_probs = F.log_softmax(p[:, :20], dim=1)
            ll = log_probs[torch.arange(p.shape[0]), sampled_aas] # line 6
            batch_ll[samples_left] += ll


        untokenized = [self.tokenizer.untokenize(s) for s in sample]
        return untokenized, batch_ll
    
