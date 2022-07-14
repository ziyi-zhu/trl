import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

config = {
    "model_name": "gpt2",
    "ref_model_name": "hakurei/lit-6B",
    "epochs": 5,
    "batch_size": 256,
    "lr": 1.41e-5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(name="run-test", project="gpt2-distil", config=config)

model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_ref = AutoModelForCausalLM.from_pretrained(config["ref_model_name"])

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

wandb.watch(model, log="all")

model.to(device)
model_ref.to(device)

ds = load_dataset(
    "ChaiML/user_model_inputs",
    split="train",
    use_auth_token="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",
)

import pdb; pdb.set_trace()

def tokenize(sample):
    sample["tokens"] = tokenizer.encode(sample["review"])[: input_size()]
    sample["query"] = tokenizer.decode(sample["tokens"])
    return sample


ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds.map(tokenize, batched=False).shuffle(seed=42)


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config["batch_size"], collate_fn=collater
)

model.train()

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(config["batch_size"]):
        gen_len = output_size()
        response = gpt2_model.generate(
            query_tensors[i].unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs
        )
        response_tensors.append(response.squeeze()[len(query_tensors[i]):])
    batch["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing["time/get_response"] = time.time() - t

    #### Compute sentiment score
    t = time.time()
    rewards = torch.tensor(
        [
            calculate_reward(q, r)
            for q, r in zip(batch["query"], batch["response"])
        ]
    ).to(device)
    timing["time/get_sentiment_preds"] = time.time() - t

    #### Run PPO step
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing["time/optimization"] = time.time() - t

    #### Log everything
    timing["time/epoch"] = time.time() - t0
    table_rows = [
        list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
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
    wandb.log(logs)
