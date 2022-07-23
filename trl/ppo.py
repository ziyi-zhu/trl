import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from transformers import DataCollatorForLanguageModeling

from .core import (
    logprobs_from_logits,
    kl_divs_from_logits,
    entropy_from_logits,
    whiten,
    clip_by_value,
    stack_dicts,
)


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
    }

    def __init__(self, model, ref_model, value_model, tokenizer, **ppo_params):
        """
        Initialize PPOTrainer.
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.ref_model = ref_model
        self.value_model = value_model
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = Adam(model.parameters(), lr=self.ppo_params["lr"])
        self.vf_optimizer = Adam(value_model.parameters(), lr=self.ppo_params["lr"])

        if self.ppo_params["adap_kl_ctrl"]:
            self.kl_ctl = AdaptiveKLController(
                self.ppo_params["init_kl_coef"],
                self.ppo_params["target"],
                self.ppo_params["horizon"],
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_params["init_kl_coef"])

        self.steps = 0

    def step(self, input_ids, attention_mask, response_mask, scores):
        """
        Run a PPO optimisation step.
        """
        labels = input_ids.roll(-1)
        label_mask = response_mask.roll(-1)

        logprobs, values, kl_divs = self.batched_forward_pass(
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        rewards = self.compute_rewards(scores, kl_divs, label_mask)

        all_stats = []
        for _ in range(self.ppo_params["ppo_epochs"]):
            batched_indices = torch.arange(self.ppo_params["batch_size"]).view(
                -1, self.ppo_params["mini_batch_size"]
            )
            for indices in batched_indices:
                train_stats = self.train_minibatch(
                    logprobs=logprobs[indices],
                    values=values[indices],
                    rewards=rewards[indices],
                    input_ids=input_ids[indices],
                    attention_mask=attention_mask[indices],
                    labels=labels[indices],
                    label_mask=label_mask[indices],
                )
                all_stats.append(train_stats)
        train_stats = stack_dicts(all_stats)

        mean_kl = kl_divs.masked_select(label_mask.bool()).mean().item()
        self.kl_ctl.update(mean_kl, self.ppo_params["batch_size"])

        stats = self.record_step_stats(
            mean_kl=mean_kl,
            train_stats=train_stats,
        )
        return stats

    @torch.no_grad()
    def batched_forward_pass(self, labels, **batch_encoded):
        """Calculate model outputs in batches."""
        logits = self.model(**batch_encoded).logits
        ref_logits = self.ref_model(**batch_encoded).logits
        values = self.value_model(**batch_encoded)

        logprobs = logprobs_from_logits(logits, labels)
        ref_logprobs = logprobs_from_logits(ref_logits, labels)
        kl_divs = kl_divs_from_logits(logits, ref_logits)
        return logprobs, values, kl_divs

    def train_minibatch(
        self, logprobs, values, rewards, labels, label_mask, **batch_encoded
    ):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs=logprobs,
            values=values,
            rewards=rewards,
            labels=labels,
            label_mask=label_mask,
            **batch_encoded,
        )

        if self.steps > 160:
            self.optimizer.zero_grad()
            loss_p.backward()
            self.optimizer.step()
        else:
            self.steps += 1

        self.vf_optimizer.zero_grad()
        loss_v.backward()
        self.vf_optimizer.step()

        return train_stats

    def compute_rewards(self, scores, kl_divs, label_mask):
        """Compute per token rewards from scores and KL-penalty."""
        last_token_mask = self.get_last_token_mask(label_mask)
        kl_penalties = -self.kl_ctl.value * kl_divs
        return kl_penalties + last_token_mask * scores.unsqueeze(-1)

    def get_last_token_mask(self, label_mask):
        last_token_mask = label_mask - label_mask.roll(-1)
        last_token_mask[last_token_mask < 0] = 0
        return last_token_mask

    def estimate_advantages(self, values, rewards, label_mask):
        next_values = (values * label_mask).roll(-1)

        delta = rewards + self.ppo_params["gamma"] * next_values - values
        response_length = label_mask.sum(-1).max().item()

        advantages = delta * label_mask
        last_advantage_estimates = advantages.clone()

        for t in range(response_length):
            last_advantage_estimates = (
                self.ppo_params["gamma"]
                * self.ppo_params["lam"]
                * last_advantage_estimates.roll(-1)
            )
            advantages += last_advantage_estimates

        return advantages * label_mask

    def value_function_loss(self, vpreds, values, returns, label_mask):
        vpreds_clipped = clip_by_value(
            vpreds,
            values - self.ppo_params["cliprange_value"],
            values + self.ppo_params["cliprange_value"],
        )

        loss_1 = ((vpreds - returns) ** 2).masked_select(label_mask.bool())
        loss_2 = ((vpreds_clipped - returns) ** 2).masked_select(label_mask.bool())
        loss = 0.5 * torch.mean(torch.max(loss_1, loss_2))

        with torch.no_grad():
            clipfrac = torch.mean(torch.gt(loss_2, loss_1).double())

        return loss, clipfrac

    def policy_gradient_loss(self, logprobs, old_logprobs, advantages, label_mask):
        ratio = torch.exp(logprobs - old_logprobs)

        loss_1 = (-advantages * ratio).masked_select(label_mask.bool())
        loss_2 = (
            -advantages
            * torch.clamp(
                ratio,
                1.0 - self.ppo_params["cliprange"],
                1.0 + self.ppo_params["cliprange"],
            )
        ).masked_select(label_mask.bool())
        loss = torch.mean(torch.max(loss_1, loss_2))

        with torch.no_grad():
            clipfrac = torch.mean(torch.gt(loss_2, loss_1).double())

        return loss, clipfrac

    def loss(self, old_logprobs, values, rewards, labels, label_mask, **batch_encoded):
        """Calculate policy and value losses."""
        advantages = self.estimate_advantages(values, rewards, label_mask)

        returns = advantages + values
        advantages = whiten(advantages, label_mask)

        logits = self.model(**batch_encoded).logits
        vpreds = self.value_model(**batch_encoded)
        logprobs = logprobs_from_logits(logits, labels)

        vf_loss, vf_clipfrac = self.value_function_loss(
            vpreds, values, returns, label_mask
        )
        pg_loss, pg_clipfrac = self.policy_gradient_loss(
            logprobs, old_logprobs, advantages, label_mask
        )

        stats = {
            "policy/loss": pg_loss.detach().cpu(),
            "policy/clipfrac": pg_clipfrac.cpu(),
            "value/loss": vf_loss.detach().cpu(),
            "value/clipfrac": vf_clipfrac.cpu(),
        }
        return pg_loss, self.ppo_params["vf_coef"] * vf_loss, stats

    def record_step_stats(self, mean_kl, train_stats):
        """Record training step statistics."""
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_coef": self.kl_ctl.value,
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        return stats
