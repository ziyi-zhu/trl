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

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer


config = {
    "run_name": str(os.environ.get("RUN_NAME", "run-test")),
    "project_name": str(os.environ.get("PROJECT_NAME", "gpt2-ppo")),
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "model_name": str(os.environ.get("MODEL_NAME", "gpt2")),
    "tokenizer_name": str(os.environ.get("TOKENIZER_NAME", "gpt2")),
    "vf_model_name": str(os.environ.get("VF_MODEL_NAME", "gpt2")),
    "ref_model_name": str(os.environ.get("REF_MODEL_NAME", "gpt2")),
    "fp16": (os.environ.get("FP16", "False") == "True"),
    "cls_model_name": str(
        os.environ.get("CLS_MODEL_NAME", "ChaiML/rewardModel90kEpoch2K1M3")
    ),
    "cls_tokenizer_name": str(
        os.environ.get("CLS_TOKENIZER_NAME", "roberta-large-mnli")
    ),
    "cls_input_size": int(os.environ.get("CLS_INPUT_SIZE", 512)),
    "cls_shift": float(os.environ.get("CLS_SHIFT", 0.0)),
    "cls_penal_coef": float(os.environ.get("CLS_PENAL_COEF", 1.2)),
    "steps": int(os.environ.get("STEPS", 50000)),
    "epochs": int(os.environ.get("EPOCHS", 5)),
    "eval_steps": int(os.environ.get("EVAL_STEPS", 10)),
    "checkpoint_steps": int(os.environ.get("CHECKPOINT_STEPS", 30)),
    "batch_size": int(os.environ.get("BATCH_SIZE", 4)),
    "eval_batch_size": int(os.environ.get("EVAL_BATCH_SIZE", 32)),
    "ppo_epochs": int(os.environ.get("PPO_EPOCHS", 4)),
    "input_size": int(os.environ.get("INPUT_SIZE", 960)),
    "output_size": int(os.environ.get("OUTPUT_SIZE", 32)),
    "lr": float(os.environ.get("LR", 1e-5)),
    "adap_kl_ctrl": (os.environ.get("ADAP_KL_CTRL", "False") == "True"),
    "init_kl_coef": float(os.environ.get("INIT_KL_COEF", 0.2)),
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


def reduce_randomness(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def tokenize(samples):
    return tokenizer(
        samples["text"],
        padding=True,
        truncation=True,
        max_length=config["input_size"],
        return_tensors="pt",
    )


def save_checkpoint(model, optimizer, steps):
    save_path = "/tmp/checkpoint-{}-state.pt".format(steps)
    print("saving checking to {}".format(save_path))
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "steps": steps,
    }
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, load_path):
    print("loading checkpoint from {}".format(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    steps = checkpoint["steps"]
    return model, optimizer, steps


def train_step(batch):
    batch_encoded = tokenize(batch).to(device)
    model_output, responses = get_model_responses(model, **batch_encoded)

    responses = [format_response(response) for response in responses]
    output_attention_mask, response_mask = get_response_mask(
        model_output, responses, **batch_encoded
    )

    response_sizes = response_mask.sum(-1)
    rewards, preds = compute_rewards(batch, responses, response_sizes)

    logs = ppo_trainer.step(
        model_output,
        output_attention_mask.to(device),
        response_mask.to(device),
        rewards,
    )

    logs.update(get_train_logs(batch, responses, rewards, preds))
    return logs


def get_response_mask(model_output, responses, input_ids, attention_mask):
    response_encoded = tokenizer(
        responses,
        padding="max_length",
        truncation=True,
        max_length=model_output.size(-1) - input_ids.size(-1),
        return_tensors="pt",
    )
    response_attention_mask = response_encoded.attention_mask.flip(-1)
    response_mask = torch.cat(
        [
            torch.zeros(attention_mask.size()),
            response_attention_mask,
        ],
        dim=-1,
    ).int()
    output_attention_mask = torch.cat(
        [
            attention_mask.cpu(),
            response_attention_mask,
        ],
        dim=-1,
    ).int()
    return output_attention_mask, response_mask


def get_train_logs(batch, responses, rewards, preds):
    logs = dict()
    logs["response_log"] = get_response_table(batch, responses, rewards, preds)
    logs["train/preds_mean"] = torch.mean(preds).cpu().numpy()
    logs["train/preds_std"] = torch.mean(preds).cpu().numpy()
    logs["train/reward_mean"] = torch.mean(rewards).cpu().numpy()
    logs["train/reward_std"] = torch.std(rewards).cpu().numpy()
    return logs


def get_response_table(batch, responses, rewards, preds):
    df_results = pd.DataFrame(
        {
            "text": batch["text"],
            "response": responses,
            "reward": rewards.cpu().tolist(),
            "pred": preds.cpu().tolist(),
        }
    )
    return wandb.Table(dataframe=df_results)


@torch.no_grad()
def compute_rewards(batch, responses, response_sizes=None):
    reward_inputs = [
        reward_input + response
        for reward_input, response in zip(batch["reward_input"], responses)
    ]
    input_encoded = reward_tokenizer(
        responses,
        padding=True,
        truncation=True,
        max_length=config["cls_input_size"],
        return_tensors="pt",
    ).to(device)
    logits = reward_model(**input_encoded).logits
    return calculate_reward_score(logits, response_sizes)


def calculate_reward_score(logits, response_sizes):
    preds = torch.softmax(logits, dim=1)
    rewards = inverse_sigmoid(preds) + config["cls_shift"]
    if response_sizes is not None:
        length_penal = torch.exp(1 - response_sizes).unsqueeze(-1)
        rewards -= config["cls_penal_coef"] * length_penal.to(device)
    return rewards[:, 1], preds[:, 1]


def inverse_sigmoid(preds):
    return torch.log(preds) - torch.log(1 - preds)


def format_response(response):
    response = clip_response_to_text(response, "\n")
    response = clip_response_to_text(response, "<|endoftext|>")
    return response


def clip_response_to_text(response, text):
    idx = response.find(text)
    return response[: idx + len(text)] if idx != -1 else response


def get_model_responses(model, input_ids, attention_mask):
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
        max_length=input_ids.size(-1) + config["output_size"]
    )
    response = output[:, input_ids.size(-1) :]
    response_decoded = tokenizer.batch_decode(response)
    return output, response_decoded


def training_loop(dataloader):
    total_ppo_steps = int(np.ceil(config["steps"] / config["batch_size"]))
    for step, batch in tqdm(zip(range(total_ppo_steps), iter(dataloader))):
        logs = train_step(batch)

        if not step % config["eval_steps"]:
            logs.update(evaluate(config["eval_batch_size"]))

        # if not step % config["checkpoint_steps"]:
        #     save_checkpoint(ppo_trainer.model, ppo_trainer.optimizer, step)

        wandb.log(logs)


@torch.no_grad()
def evaluate(batch_size):
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"], batch_size=batch_size
    )
    eval_batch = next(iter(eval_dataloader))
    return evaluate_step(eval_batch)


def evaluate_step(batch):
    batch_encoded = tokenize(batch).to(device)
    _, responses = get_model_responses(model, **batch_encoded)
    _, ref_responses = get_model_responses(model_ref, **batch_encoded)

    responses = [format_response(response) for response in responses]
    ref_responses = [format_response(response) for response in ref_responses]

    rewards, preds = compute_rewards(batch, responses)
    ref_rewards, ref_preds = compute_rewards(batch, ref_responses)

    logs = dict()

    df_results = pd.DataFrame(
        {
            "text": batch["text"],
            "ppo_response": responses,
            "original_response": ref_responses,
            "ppo_reward": rewards.cpu().tolist(),
            "ppo_pred": preds.cpu().tolist(),
            "original_reward": ref_rewards.cpu().tolist(),
            "original_pred": ref_preds.cpu().tolist(),
        }
    )
    logs["evaluation/comparison_log"] = wandb.Table(dataframe=df_results)

    logs["evaluation/original_pred_mean"] = torch.mean(ref_preds).cpu().numpy()
    logs["evaluation/original_reward_mean"] = torch.mean(ref_rewards).cpu().numpy()
    logs["evaluation/ppo_pred_mean"] = torch.mean(preds).cpu().numpy()
    logs["evaluation/ppo_reward_mean"] = torch.mean(rewards).cpu().numpy()

    return logs


reduce_randomness(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"], use_auth_token=config["auth_token"]
)
model_ref = AutoModelForCausalLM.from_pretrained(
    config["ref_model_name"], use_auth_token=config["auth_token"]
)

tokenizer = AutoTokenizer.from_pretrained(
    config["tokenizer_name"],
    truncation_size="left",
    padding_side="left",
)
tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "temperature": config["temperature"],
    "top_k": config["top_k"],
    "top_p": config["top_p"],
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

value_model = GPT2HeadWithValueModel.from_pretrained(config["vf_model_name"]).to(device)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "lvwerra/distilbert-imdb", use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    "lvwerra/distilbert-imdb", truncation_side="left", padding_side="left"
)

ppo_trainer = PPOTrainer(model, model_ref, value_model, tokenizer, **config)


if __name__ == "__main__":
    dataset = load_dataset(
        "ChaiML/user_model_inputs", use_auth_token=config["auth_token"]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=config["batch_size"]
    )

    wandb.login(key=config["wandb_key"])
    wandb.init(name=config["run_name"], project="batch-debug", config=config)
    wandb.watch(model, log="all")

    training_loop(dataloader)
