"""Pretraining objectives and masking policy for foundation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from src.foundation_input.types import SPECIAL_TOKEN_IDS, FoundationBatch

if TYPE_CHECKING:
    from .encoder import FoundationModelConfig, FoundationOutput


@dataclass(frozen=True)
class PretrainObjectivesConfig:
    """Weights and parameters for multi-task pretraining objectives."""

    mask_prob: float = 0.15
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_token_prob: float = 0.1
    masked_categorical_weight: float = 1.0
    masked_numeric_weight: float = 1.0
    next_action_weight: float = 0.5
    next_token_weight: float | None = None
    unmaskable_key_ids: tuple[int, ...] = ()


class MaskingPolicy:
    """Applies deterministic or stochastic masking to FoundationBatch tokens.

    Guarantees special tokens PAD (0), CLS (1), and SEP (2) are never masked.
    """

    def __init__(self, config: PretrainObjectivesConfig) -> None:
        self.config = config

    def apply_mask(
        self,
        batch: FoundationBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[FoundationBatch, dict[str, torch.Tensor]]:
        """Mask tokens in a batch and return the masked batch plus loss targets."""
        device = batch.input_ids.device
        batch_size, seq_len = batch.input_ids.shape

        # Candidates: valid tokens excluding PAD, CLS, SEP
        special_ids = {
            SPECIAL_TOKEN_IDS["PAD"],
            SPECIAL_TOKEN_IDS["CLS"],
            SPECIAL_TOKEN_IDS["SEP"],
        }
        is_special = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for special_id in special_ids:
            is_special |= batch.input_ids.eq(special_id)

        candidate_mask = batch.attention_mask & (~is_special)
        for key_id in self.config.unmaskable_key_ids:
            candidate_mask &= batch.key_ids.ne(key_id)

        # Random Bernoulli mask
        rand_probs = torch.rand((batch_size, seq_len), generator=generator, device=device)
        should_mask = candidate_mask & (rand_probs < self.config.mask_prob)

        # For any sequence with candidates but zero masked tokens, force-mask at least one
        has_candidates = candidate_mask.any(dim=1)
        none_masked = (~should_mask.any(dim=1)) & has_candidates
        if none_masked.any():
            for row_idx in torch.nonzero(none_masked).squeeze(1):
                cand_indices = torch.nonzero(candidate_mask[row_idx]).squeeze(1)
                if len(cand_indices) > 0:
                    chosen = cand_indices[torch.randint(0, len(cand_indices), (1,), generator=generator)]
                    should_mask[row_idx, chosen] = True

        # Copy tensors for masking
        masked_input_ids = batch.input_ids.clone()
        masked_numeric_values = batch.numeric_values.clone()
        masked_numeric_missing = batch.numeric_missing_mask.clone()

        # Decide replacement: [MASK] (80%), Random token (10%), or Keep original (10%)
        replace_rand = torch.rand((batch_size, seq_len), generator=generator, device=device)
        mask_token_mask = should_mask & (replace_rand < self.config.mask_token_prob)
        random_token_mask = (
            should_mask
            & (replace_rand >= self.config.mask_token_prob)
            & (replace_rand < (self.config.mask_token_prob + self.config.random_token_prob))
        )

        # Replace with MASK token (ID 3)
        masked_input_ids[mask_token_mask] = SPECIAL_TOKEN_IDS["MASK"]

        # Replace with random token (from vocab range [5, max_val])
        if random_token_mask.any():
            random_tokens = torch.randint(
                SPECIAL_TOKEN_IDS["UNK"] + 1,
                max(SPECIAL_TOKEN_IDS["UNK"] + 2, batch.input_ids.max().item() + 1),
                (batch_size, seq_len),
                generator=generator,
                device=device,
            )
            masked_input_ids[random_token_mask] = random_tokens[random_token_mask]

        # For numeric values at masked positions, zero them out and mark missing
        masked_numeric_values[should_mask] = 0.0
        masked_numeric_missing[should_mask] = True

        masked_batch = FoundationBatch(
            input_ids=masked_input_ids,
            key_ids=batch.key_ids,
            modality_ids=batch.modality_ids,
            source_ids=batch.source_ids,
            position_ids=batch.position_ids,
            time_gap_ids=batch.time_gap_ids,
            attention_mask=batch.attention_mask,
            numeric_values=masked_numeric_values,
            numeric_missing_mask=masked_numeric_missing,
            user_ids=batch.user_ids,
            targets=batch.targets,
        )

        targets = {
            "masked_positions": should_mask,
            "target_input_ids": batch.input_ids.clone(),
            "target_numeric_values": batch.numeric_values.clone(),
            "target_modality_ids": batch.modality_ids.clone(),
            "target_numeric_missing": batch.numeric_missing_mask.clone(),
        }
        return masked_batch, targets


class PretrainObjectives(nn.Module):
    """Multi-task objective heads and loss computation for foundation pretraining."""

    def __init__(
        self,
        model_config: FoundationModelConfig,
        objectives_config: PretrainObjectivesConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.objectives_config = objectives_config or PretrainObjectivesConfig()
        self.masking_policy = MaskingPolicy(self.objectives_config)

        # Head 1: Categorical / ID token prediction over value vocabulary
        self.categorical_head = nn.Linear(
            model_config.d_model, model_config.value_vocab_size, bias=True
        )

        # Head 2: Numeric value regression
        self.numeric_head = nn.Sequential(
            nn.Linear(model_config.d_model, model_config.d_model),
            nn.GELU(),
            nn.Linear(model_config.d_model, 1),
        )

        # Head 3: Next action prediction from short-term user embedding
        self.next_action_head = nn.Linear(
            model_config.d_model, model_config.value_vocab_size, bias=True
        )

    def forward(
        self,
        output: FoundationOutput,
        targets: dict[str, torch.Tensor],
        batch_targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction and prediction losses.

        Args:
            output: FoundationOutput with hidden_states and user embeddings.
            targets: Targets created by MaskingPolicy.apply_mask.
            batch_targets: Optional targets from FoundationBatch.targets (e.g. next_action).

        Returns:
            Dictionary with 'loss', component losses, and monitoring metrics.
        """
        device = output.hidden_states.device
        masked_positions = targets["masked_positions"]
        target_modality_ids = targets["target_modality_ids"]
        is_numeric = target_modality_ids.eq(self.model_config.numeric_modality_id)

        # 1. Masked Categorical Loss (CE over value vocabulary)
        masked_cat_mask = masked_positions & (~is_numeric)
        cat_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        cat_correct = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_cat_masked = masked_cat_mask.sum().item()

        if num_cat_masked > 0:
            cat_logits = self.categorical_head(output.hidden_states[masked_cat_mask])
            cat_labels = targets["target_input_ids"][masked_cat_mask]
            cat_loss = F.cross_entropy(cat_logits, cat_labels)
            cat_correct = (cat_logits.argmax(dim=-1) == cat_labels).float().mean()

        # 2. Masked Numeric Loss (MSE on normalized numeric values)
        masked_num_mask = (
            masked_positions
            & is_numeric
            & (~targets["target_numeric_missing"])
        )
        num_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_numeric_masked = masked_num_mask.sum().item()

        if num_numeric_masked > 0:
            predicted_num = self.numeric_head(output.hidden_states[masked_num_mask]).squeeze(-1)
            target_num = targets["target_numeric_values"][masked_num_mask]
            num_loss = F.mse_loss(predicted_num, target_num)

        # 3. Next Action / Next Token Prediction Loss (from output.short_term)
        next_action_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        next_action_correct = torch.tensor(0.0, device=device, dtype=torch.float32)

        next_token_target = None
        if batch_targets is not None:
            if "next_token" in batch_targets:
                next_token_target = batch_targets["next_token"]
            elif "next_action_type_ids" in batch_targets:
                next_token_target = batch_targets["next_action_type_ids"]

        if next_token_target is not None:
            valid_mask = batch_targets.get(
                "next_action_valid_mask",
                batch_targets.get(
                    "next_token_valid_mask",
                    torch.ones_like(next_token_target, dtype=torch.bool),
                ),
            )
            # Only compute if there are valid targets and target ID > 0
            valid_targets = valid_mask & (next_token_target > 0)
            if valid_targets.any():
                action_logits = self.next_action_head(output.short_term[valid_targets])
                action_labels = next_token_target[valid_targets]
                next_action_loss = F.cross_entropy(action_logits, action_labels)
                next_action_correct = (action_logits.argmax(dim=-1) == action_labels).float().mean()

        # Total weighted loss
        action_weight = (
            self.objectives_config.next_token_weight
            if getattr(self.objectives_config, "next_token_weight", None) is not None
            else self.objectives_config.next_action_weight
        )
        total_loss = (
            self.objectives_config.masked_categorical_weight * cat_loss
            + self.objectives_config.masked_numeric_weight * num_loss
            + action_weight * next_action_loss
        )

        return {
            "loss": total_loss,
            "masked_loss": cat_loss + num_loss,
            "masked_categorical_loss": cat_loss,
            "masked_categorical_accuracy": cat_correct,
            "masked_numeric_loss": num_loss,
            "next_token_loss": next_action_loss,
            "next_token_accuracy": next_action_correct,
            "next_action_loss": next_action_loss,
            "next_action_accuracy": next_action_correct,
            "num_masked_tokens": torch.tensor(num_cat_masked + num_numeric_masked, device=device),
        }
