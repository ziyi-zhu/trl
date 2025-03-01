import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch


def reduce_randomness(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


reduce_randomness(42)

config = {
    "run_name": str(os.environ.get("RUN_NAME", "run-test")),
    "project_name": str(os.environ.get("PROJECT_NAME", "gpt2-ppo")),
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "model_name": "ChaiML/distil-gpt2-ppo-v1",
    "tokenizer_name": str(os.environ.get("TOKENIZER_NAME", "gpt2")),
    "vf_model_name": str(os.environ.get("VF_MODEL_NAME", "gpt2")),
    "ref_model_name": "hakurei/lit-6B",
    "cls_model_name": str(
        os.environ.get("CLS_MODEL_NAME", "ChaiML/rewardModel90kEpoch2K1M3")
    ),
    "cls_tokenizer_name": str(
        os.environ.get("CLS_TOKENIZER_NAME", "roberta-large-mnli")
    ),
    "cls_shift": float(os.environ.get("CLS_SHIFT", -3.0)),
    "cls_penal_coef": float(os.environ.get("CLS_PENAL_COEF", 1.2)),
    "steps": int(os.environ.get("STEPS", 50000)),
    "epochs": int(os.environ.get("EPOCHS", 5)),
    "eval_steps": int(os.environ.get("EVAL_STEPS", 10)),
    "checkpoint_steps": int(os.environ.get("CHECKPOINT_STEPS", 100)),
    "batch_size": int(os.environ.get("BATCH_SIZE", 32)),
    "forward_batch_size": int(os.environ.get("FORWARD_BATCH_SIZE", 16)),
    "ppo_epochs": int(os.environ.get("PPO_EPOCHS", 4)),
    "input_size": int(os.environ.get("INPUT_SIZE", 960)),
    "output_size": int(os.environ.get("OUTPUT_SIZE", 32)),
    "lr": float(os.environ.get("LR", 1e-5)),
    "adap_kl_ctrl": (os.environ.get("ADAP_KL_CTRL", "False") == "True"),
    "init_kl_coef": float(os.environ.get("INIT_KL_COEF", 0.05)),
    "target": int(os.environ.get("TARGET", 6)),
    "horizon": int(os.environ.get("HORIZON", 10000)),
    "gamma": float(os.environ.get("GAMMA", 1.0)),
    "lam": float(os.environ.get("LAM", 0.95)),
    "cliprange": float(os.environ.get("CLIPRANGE", 0.2)),
    "cliprange_value": float(os.environ.get("CLIPRANGE_VALUE", 0.2)),
    "init_steps": int(os.environ.get("INIT_STEPS", 0)),
    "vf_coef": float(os.environ.get("VF_COEF", 0.1)),
    "temperature": float(os.environ.get("TEMPERATURE", 1.0)),
    "top_k": int(os.environ.get("TOP_K", 0)),
    "top_p": float(os.environ.get("TOP_P", 1.0)),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset(
    "ChaiML/user_model_inputs", split="train", use_auth_token=config["auth_token"]
)

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"], use_auth_token=config["auth_token"]
)
model_ref = AutoModelForCausalLM.from_pretrained(
    config["ref_model_name"], use_auth_token=config["auth_token"]
).half()

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "temperature": 0.72,
    "repetition_penalty": 1.13125,
    "top_k": 0,
    "top_p": 0.725,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

reward_model = AutoModelForSequenceClassification.from_pretrained(
    config["cls_model_name"], use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    config["cls_tokenizer_name"], truncation_side="left", padding_side="left"
)


def tokenize(sample):
    sample["tokens"] = tokenizer.encode(sample["text"])[-config["input_size"] :]
    sample["query"] = tokenizer.decode(sample["tokens"])
    return sample


# ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds.map(tokenize, batched=False).shuffle(seed=42)


def calculate_reward(query, response, response_len, return_preds=False):
    encoded_input = reward_tokenizer(
        query + response, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    logits = reward_model(**encoded_input).logits
    preds = torch.softmax(logits, dim=1)
    rewards = shifted_logits_with_penalty(inverse_sigmoid(preds), response_len)

    if return_preds:
        return rewards[0, 1], preds[0, 1]
    else:
        return rewards[0, 1]


def inverse_sigmoid(preds):
    return torch.log(preds) - torch.log(1 - preds)


def shifted_logits_with_penalty(logits, response_len):
    return (
        logits
        + config["cls_shift"]
        - config["cls_penal_coef"] * np.exp(1 - response_len)
    )


def evaluate(eval_batch):
    game_data = dict()
    game_data["query"] = eval_batch["query"]
    query_tensors = [torch.tensor(t).long().to(device) for t in eval_batch["tokens"]]

    model.eval()

    #### get response from gpt2 and gpt2_ref
    response_tensors_ref, response_tensors = [], []
    for i in range(len(query_tensors)):
        query_len = len(query_tensors[i])

        output_ref = model_ref.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors_ref.append(clip_response(output_ref, query_len))

        output = model.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors.append(clip_response(output, query_len))

    #### decode responses
    game_data["original_model_response"] = [
        tokenizer.decode(r) for r in response_tensors_ref
    ]
    game_data["rl_model_response"] = [tokenizer.decode(r) for r in response_tensors]

    # responses using original model
    rewards = torch.tensor(
        [
            calculate_reward(q, r, len(rt), return_preds=True)
            for q, r, rt in zip(
                eval_batch["reward_input"],
                game_data["original_model_response"],
                response_tensors_ref,
            )
        ]
    )
    game_data["original_model_rewards"] = rewards[:, 0]
    game_data["original_model_preds"] = rewards[:, 1]

    # responses using new RL model
    rewards = torch.tensor(
        [
            calculate_reward(q, r, len(rt), return_preds=True)
            for q, r, rt in zip(
                eval_batch["reward_input"],
                game_data["rl_model_response"],
                response_tensors,
            )
        ]
    )
    game_data["rl_model_rewards"] = rewards[:, 0]
    game_data["rl_model_preds"] = rewards[:, 1]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    logs = dict()
    logs.update({"evaluation/comparison_table": wandb.Table(dataframe=df_results)})

    # update rewards and preds how they change over time
    mean_reward_before = torch.mean(game_data["original_model_rewards"])
    mean_preds_before = torch.mean(game_data["original_model_preds"])

    mean_reward_after = torch.mean(game_data["rl_model_rewards"])
    mean_preds_after = torch.mean(game_data["rl_model_preds"])

    logs.update(
        {
            "evaluation/original_model_mean_reward": mean_reward_before.cpu().numpy(),
            "evaluation/original_model_mean_preds": mean_preds_before.cpu().numpy(),
            "evaluation/rl_model_mean_reward": mean_reward_after.cpu().numpy(),
            "evaluation/rl_model_mean_preds": mean_preds_after.cpu().numpy(),
        }
    )

    return logs


def clip_response(response, query_len):
    response = response[query_len:]
    stop_idx = (response == torch.tensor(198)).nonzero().flatten()
    if len(stop_idx) > 0:
        response = response[: stop_idx[0] + 1]
    return response


logs = evaluate(ds[:256])
