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
    whiten,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    average_torch_dicts,
    stats_to_np,
    stack_dicts,
    add_suffix,
    WANDB_PADDING,
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

    def step(self, input_ids, attention_mask, response_mask, scores):
        """
        Run a PPO optimisation step.
        """
        logprobs, ref_logprobs, values = self.batched_forward_pass(input_ids, attention_mask)
        rewards, kl = self.compute_rewards(scores, logprobs, ref_logprobs, response_mask)

        for _ in range(self.ppo_params["ppo_epochs"]):
            loss_p, loss_v, entropy = self.train_minibatch(logprobs, values, rewards, input_ids, attention_mask, response_mask)

        label_mask = response_mask.roll(-1)
        mean_kl = torch.mean(kl.masked_select(label_mask.bool())).item()

        self.kl_ctl.update(mean_kl, self.ppo_params["batch_size"])

        stats = self.record_step_stats(
            mean_kl=mean_kl,
        )
        return stats

    @torch.no_grad()
    def batched_forward_pass(self, input_ids, attention_mask):
        """Calculate model outputs in batches."""
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits
        values = self.value_model(input_ids, attention_mask=attention_mask)
        
        labels = input_ids.roll(-1)
        logprobs = logprobs_from_logits(logits, labels)
        ref_logprobs = logprobs_from_logits(ref_logits, labels)
        return logprobs, ref_logprobs, values

    def train_minibatch(self, logprobs, values, rewards, input_ids, attention_mask, response_mask):
        """Train one PPO minibatch"""
        loss_p, loss_v, entropy = self.loss(
            logprobs, values, rewards, input_ids, attention_mask, response_mask
        )

        self.optimizer.zero_grad()
        loss_p.backward()
        self.optimizer.step()

        self.vf_optimizer.zero_grad()
        loss_v.backward()
        self.vf_optimizer.step()

        return loss_p, loss_v, entropy

    def compute_rewards(self, scores, logprobs, ref_logprobs, response_mask):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        kl_penalties = -self.kl_ctl.value * kl

        label_mask = response_mask.roll(-1)
        last_token_mask = self.get_last_token_mask(label_mask)
        
        rewards = (kl_penalties + last_token_mask * scores.unsqueeze(-1))
        return rewards, kl


    def get_last_token_mask(self, label_mask):
        last_token_mask = label_mask - label_mask.roll(-1)
        last_token_mask[last_token_mask < 0] = 0
        return last_token_mask


    def estimate_advantages(self, values, rewards, response_mask):
        next_values = values.roll(-1)
        delta = rewards + self.ppo_params["gamma"] * next_values - values

        label_mask = response_mask.roll(-1)
        response_length = response_mask.sum(-1).max().item()

        advantages = delta * label_mask
        last_advantage_estimates = advantages.clone()

        for t in range(response_length):
            last_advantage_estimates = self.ppo_params["gamma"] * self.ppo_params["lam"] * last_advantage_estimates.roll(-1)
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
        clipfrac = torch.mean(torch.gt(loss_2, loss_1).double())

        return loss, clipfrac

    def policy_gradient_loss(self, logprobs, old_logprobs, advantages, label_mask):
        ratio = torch.exp(logprobs - old_logprobs)

        loss_1 = (-advantages * ratio).masked_select(label_mask.bool())
        loss_2 = (-advantages * torch.clamp(
            ratio,
            1.0 - self.ppo_params["cliprange"],
            1.0 + self.ppo_params["cliprange"],
        )).masked_select(label_mask.bool())
        loss = torch.mean(torch.max(loss_1, loss_2))
        clipfrac = torch.mean(torch.gt(loss_2, loss_1).double())

        return loss, clipfrac

    def loss(self, old_logprobs, values, rewards, input_ids, attention_mask, response_mask):
        """Calculate policy and value losses."""
        advantages = self.estimate_advantages(values, rewards, response_mask)

        label_mask = response_mask.roll(-1)
        returns = advantages + values
        advantages = whiten(advantages, label_mask)

        logits = self.model(input_ids, attention_mask=attention_mask).logits
        vpreds = self.value_model(input_ids, attention_mask=attention_mask)

        labels = input_ids.roll(-1)
        logprobs = logprobs_from_logits(logits, labels)

        vf_loss, vf_clipfrac = self.value_function_loss(vpreds, values, returns, label_mask)
        pg_loss, pg_clipfrac = self.policy_gradient_loss(logprobs, old_logprobs, advantages, label_mask)

        entropy = torch.mean(entropy_from_logits(logits))
        return pg_loss, vf_loss, entropy

    def record_step_stats(self, mean_kl):
        """Record training step statistics."""
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_coef": self.kl_ctl.value,
        }
        return stats
