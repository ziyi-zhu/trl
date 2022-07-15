import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

config = {
    "model_name": "gpt2",
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "ref_model_name": "hakurei/lit-6B",
    "cls_model_name": "ChaiML/rewardModel90kEpoch2K1M3",
    "cls_tokenizer_name": "roberta-large-mnli",
    "epochs": 5,
    "batch_size": 8,
    "eval_interval": 128,
    "lr": 1e-6,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key=config["wandb_key"])
wandb.init(name="run-test", project="gpt2-distil", config=config)

ds = load_dataset(
    "ChaiML/user_model_inputs",
    use_auth_token=config["auth_token"],
)

model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_ref = AutoModelForCausalLM.from_pretrained(config["ref_model_name"]).half()

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

gen_kwargs = {
    "min_length": -1,
    "temperature": 1.0,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

wandb.watch(model, log="all")

model.to(device)
model_ref.to(device)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    config["cls_model_name"], use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    config["cls_tokenizer_name"], truncation_side="left", padding_side="left"
)


def tokenize(samples):
    return tokenizer(samples["text"], max_length=1024, truncation=True, padding="max_length")


ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds.map(tokenize, batched=True).shuffle(seed=42)

ds.set_format(type="torch", columns=["input_ids", "attention_mask", "text", "reward_input"])

train_dataloader = torch.utils.data.DataLoader(
    ds["train"], batch_size=config["batch_size"]
)
valid_dataloader = torch.utils.data.DataLoader(
    ds["validation"], batch_size=config["batch_size"]
)

cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
kl_div = torch.nn.KLDivLoss(reduction="none")

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

total_steps = len(train_dataloader) * config["epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def train_step(batch):
    logs = dict()

    model.train()
    batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

    with torch.no_grad():
        target_logits = model_ref(**batch).logits[:, :, : tokenizer.vocab_size]
        probs = torch.softmax(target_logits, dim=-1)

    model.zero_grad()

    mask = batch["attention_mask"].flatten().bool()
    logits = model(**batch).logits[:, :, : tokenizer.vocab_size]

    loss = cross_entropy(
        logits.flatten(end_dim=1), probs.flatten(end_dim=1)
    ).masked_select(mask).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    kl_loss = kl_div(
        logits.flatten(end_dim=1), probs.flatten(end_dim=1)
    ).sum(dim=-1).masked_select(mask).mean()

    logs["objective/cross_entropy"] = loss.item()
    logs["objective/kl_divergence"] = kl_loss.item()

    return loss, logs


def validation_step(batch):
    model.eval()
    batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

    with torch.no_grad():
        target_logits = model_ref(**batch).logits[:, :, : tokenizer.vocab_size]
        probs = torch.softmax(target_logits, dim=-1)

        mask = batch["attention_mask"].flatten().bool()
        logits = model(**batch).logits[:, :, : tokenizer.vocab_size]

        loss = cross_entropy(
            logits.flatten(end_dim=1), probs.flatten(end_dim=1)
        ).masked_select(mask)

    return loss


def evaluation():
    logs, table = dict(), dict()
    valid_loss = []

    for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):
        loss = validation_step(batch)
        valid_loss.append(loss.item())

    logs["loss/validation"] = np.mean(valid_loss)

    eval_batch = next(iter(valid_dataloader))

    table["text"] = eval_batch["text"]
    input_ids = eval_batch["input_ids"]

    model.eval()

    response_tensors_ref, response_tensors = [], []
    for i in range(len(input_ids)):
        query_len = len(input_ids[i])

        output_ref = model_ref.generate(
            input_ids[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors_ref.append(clip_response(output_ref, query_len))

        output = model.generate(
            input_ids[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors.append(clip_response(output, query_len))

    table["original_model_response"] = [
        tokenizer.decode(r) for r in response_tensors_ref
    ]
    table["distil_model_response"] = [tokenizer.decode(r) for r in response_tensors]

    rewards = torch.tensor(
        [
            calculate_reward(q, r)
            for q, r in zip(
                eval_batch["reward_input"],
                table["original_model_response"],
            )
        ]
    )
    table["original_model_rewards"] = rewards

    rewards = torch.tensor(
        [
            calculate_reward(q, r)
            for q, r in zip(
                eval_batch["reward_input"],
                table["distil_model_response"],
            )
        ]
    )
    table["distil_model_rewards"] = rewards

    df_results = pd.DataFrame(table)

    logs.update({"evaluation/comparison_table": wandb.Table(dataframe=df_results)})

    mean_reward_before = torch.mean(table["original_model_rewards"])
    mean_reward_after = torch.mean(table["distil_model_rewards"])

    logs.update(
        {
            "evaluation/original_model_mean_reward": mean_reward_before.cpu().numpy(),
            "evaluation/distil_model_mean_reward": mean_reward_after.cpu().numpy(),
        }
    )

    return logs


def calculate_reward(query, response):
    encoded_input = reward_tokenizer(
        query + response, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    logits = reward_model(**encoded_input).logits
    preds = torch.softmax(logits, dim=1)
    return preds[0, 1]


def clip_response(response, query_len):
    response = response[query_len:]
    stop_idx = (response == torch.tensor(198)).nonzero().flatten()
    if len(stop_idx) > 0:
        response = response[: stop_idx[0] + 1]
    return response


for epoch in range(config["epochs"]):
    print("======== Epoch {:} / {:} ========".format(epoch + 1, config["epochs"]))

    train_loss = []

    for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        logs, timing = dict(), dict()
        t0 = time.time()

        loss, stats = train_step(batch)
        train_loss.append(loss.item())

        logs.update(stats)

        timing["time/batch"] = time.time() - t0
        logs.update(timing)

        if not (step + 1) % config["eval_interval"]:
            logs.update(evaluation())
            logs["loss/train"] = np.mean(train_loss[-config["eval_interval"]:])

        wandb.log(logs)
