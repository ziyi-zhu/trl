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
    "cls_shift": float(os.environ.get("CLS_SHIFT", -3.0)),
    "cls_penal_coef": float(os.environ.get("CLS_PENAL_COEF", 1.2)),
    "steps": int(os.environ.get("STEPS", 50000)),
    "epochs": int(os.environ.get("EPOCHS", 5)),
    "eval_steps": int(os.environ.get("EVAL_STEPS", 10)),
    "checkpoint_steps": int(os.environ.get("CHECKPOINT_STEPS", 30)),
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

#wandb.login(key=config["wandb_key"])
#wandb.init(name=config["run_name"], project=config["project_name"], config=config)

ds = load_dataset(
    "ChaiML/user_model_inputs", split="train", use_auth_token=config["auth_token"]
)

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

#wandb.watch(model, log="all")

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
    config["cls_model_name"], use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    config["cls_tokenizer_name"], truncation_side="left", padding_side="left"
)


def tokenize(samples):
    return tokenizer(samples["text"], padding=True, truncation=True, max_length=config["input_size"], return_tensors="pt")


dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config["batch_size"]
)

ppo_trainer = PPOTrainer(model, model_ref, value_model, tokenizer, **config)

total_ppo_steps = int(np.ceil(config["steps"] / config["batch_size"]))
total_epochs = config["epochs"]

dataloader_iter = iter(dataloader)
eval_batch = dataloader_iter.next()


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
    responses, responses_decoded = get_model_responses(model, **batch_encoded)
    rewards, preds = compute_rewards(batch, responses_decoded)
    import pdb; pdb.set_trace()
    stats = ppo_trainer.step(batch_encoded, responses, rewards)


def compute_rewards(batch, responses):
    reward_inputs = [reward_input + format_response(response) for reward_input, response in zip(batch["reward_input"], responses)]
    input_encoded = reward_tokenizer(reward_inputs, padding=True, truncation=True, max_length=config["cls_input_size"], return_tensors="pt").to(device)
    logits = reward_model(**input_encoded).logits
    return calculate_reward_score(logits)


def calculate_reward_score(logits):
    preds = torch.softmax(logits, dim=1)
    rewards = shifted_logits_with_penalty(inverse_sigmoid(preds))
    return rewards[:, 1], preds[:, 1]


def shifted_logits_with_penalty(logits):
    return (
        logits
        + config["cls_shift"]
#        - config["cls_penal_coef"] * np.exp(1 - response_len)
    )


def inverse_sigmoid(preds):
    return torch.log(preds) - torch.log(1 - preds)


def format_response(response):
    response = clip_response_to_text(response, "\n")
    response = clip_response_to_text(response, "<|endoftext|>")
    print(response)
    return response


def clip_response_to_text(response, text):
    idx = response.find(text)
    return response[:idx] if idx != -1 else response


def get_model_responses(model, input_ids, attention_mask):
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs, max_length=input_ids.size(-1) + config["output_size"])
    response = output[:, input_ids.size(-1):]
    response_decoded = tokenizer.batch_decode(response)
    return response, response_decoded


for epoch in range(total_epochs):
    print(f"Epoch {epoch + 1}/{total_epochs}")

    for step, batch in tqdm(zip(range(total_ppo_steps), iter(dataloader))):
        logs, timing = dict(), dict()
        t0 = time.time()

        train_step(batch)

        query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

        model.eval()

        #### Get response from gpt2
        t = time.time()
        response_tensors = []
        for i in range(len(query_tensors)):
            query_len = len(query_tensors[i])
            response = model.generate(
                query_tensors[i].unsqueeze(dim=0),
                max_length=query_len + config["output_size"],
                **gen_kwargs,
            ).squeeze()
            response_tensors.append(clip_response(response, query_len))

        batch["response"] = [tokenizer.decode(r) for r in response_tensors]
        timing["time/get_response"] = time.time() - t

        #### Compute reward score
        t = time.time()
        rewards = torch.tensor(
            [
                calculate_reward(q, r, len(rt))
                for q, r, rt in zip(
                    batch["reward_input"], batch["response"], response_tensors
                )
            ]
        ).to(device)
        timing["time/get_reward_preds"] = time.time() - t

        #### Run PPO step
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing["time/optimization"] = time.time() - t

        #### Log everything
        timing["time/epoch"] = time.time() - t0
        table_rows = [
            list(r)
            for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
        ]
        logs.update(
            {
                "game_log": wandb.Table(
                    columns=["query", "response", "reward"], rows=table_rows
                )
            }
        )
        logs.update(timing)
        logs.update(stats)
        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
        logs["env/reward_dist"] = rewards.cpu().numpy()

        for key in logs:
            if isinstance(logs[key], list):
                if isinstance(logs[key][0], torch.Tensor):
                    logs[key] = [array.cpu().numpy() for array in logs[key]]

        if not step % config["eval_steps"]:
            logs.update(evaluate(eval_batch))

        if not step % config["checkpoint_steps"]:
            save_checkpoint(ppo_trainer.model, ppo_trainer.optimizer, step)

        wandb.log(logs)
