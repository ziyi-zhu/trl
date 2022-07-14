import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

config = {
    "model_name": "gpt2",
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "ref_model_name": "hakurei/lit-6B",
    "epochs": 10,
    "batch_size": 8,
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

wandb.watch(model, log="all")

model.to(device)
model_ref.to(device)


def tokenize(samples):
    return tokenizer(samples["text"], max_length=1024, truncation=True, padding="max_length")


# ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds.map(tokenize, batched=True).shuffle(seed=42)

ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

train_dataloader = torch.utils.data.DataLoader(
    ds["train"], batch_size=config["batch_size"]
)
valid_dataloader = torch.utils.data.DataLoader(
    ds["validation"], batch_size=config["batch_size"]
)

cross_entropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

total_steps = len(train_dataloader) * config["epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def train_step(batch):
    model.train()
    batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

    with torch.no_grad():
        targets = model_ref(**batch).logits
        probs = torch.softmax(targets[:, :, : tokenizer.vocab_size], dim=-1)

    model.zero_grad()

    mask = batch["attention_mask"].flatten().bool()
    logits = model(**batch).logits
    loss = cross_entropy(
        logits.flatten(end_dim=1)[mask], probs.flatten(end_dim=1)[mask]
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    return loss


def validation_step(batch):
    model.eval()
    batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

    with torch.no_grad():
        targets = model_ref(**batch).logits
        probs = torch.softmax(targets[:, :, : tokenizer.vocab_size], dim=-1)

        mask = batch["attention_mask"].flatten().bool()
        logits = model(**batch).logits
        loss = cross_entropy(
            logits.flatten(end_dim=1)[mask], probs.flatten(end_dim=1)[mask]
        )

    return loss


def evaluation():
    valid_loss = 0

    for batch in tqdm(valid_dataloader, total=len(valid_dataloader)):
        loss = validation_step(batch)
        valid_loss += loss.item()

    return valid_loss / len(valid_dataloader)


for epoch in range(config["epochs"]):
    print("======== Epoch {:} / {:} ========".format(epoch + 1, config["epochs"]))

    for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        logs, timing = dict(), dict()
        t0 = time.time()

        loss = train_step(batch)

        timing["time/batch"] = time.time() - t0
        logs.update(timing)

        logs["loss/train"] = loss.item()

        if not step % 100:
            logs["loss/validation"] = evaluation()

        wandb.log(logs)
