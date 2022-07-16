import torch
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
    "input_size": 960,
    "output_size": 32,
    "eval_interval": 128,
    "lr": 1e-5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("ChaiML/user_model_inputs", use_auth_token=config["auth_token"])

model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_ref = AutoModelForCausalLM.from_pretrained(config["ref_model_name"]).half()

tokenizer = AutoTokenizer.from_pretrained(config["model_name"], truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token

gen_kwargs = {
    "min_length": -1,
    "temperature": 0.8,
    "top_k": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

model.to(device)
model_ref.to(device)


def tokenize(samples):
    return tokenizer(
        samples["text"], max_length=1024, truncation=True, padding="max_length"
    )


# ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds["validation"].map(tokenize, batched=True).shuffle(seed=42)

eval_batch = ds[:64]

ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

valid_dataloader = torch.utils.data.DataLoader(ds, batch_size=config["batch_size"])

cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
kl_div = torch.nn.KLDivLoss(reduction="none")

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


def validation_step(batch):
    model.eval()
    batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

    with torch.no_grad():
        target_logits = model_ref(**batch).logits[:, :, : tokenizer.vocab_size]
        probs = torch.softmax(target_logits, dim=-1)

        mask = batch["attention_mask"].flatten().bool()
        logits = model(**batch).logits[:, :, : tokenizer.vocab_size]

        loss = (
            cross_entropy(logits.flatten(end_dim=1), probs.flatten(end_dim=1))
            .masked_select(mask)
            .mean()
        )

    return loss


def evaluation():
    logs, table = dict(), dict()
    valid_loss = []

    for batch in tqdm(valid_dataloader):
        loss = validation_step(batch)
        valid_loss.append(loss.item())

    logs["loss/validation"] = np.mean(valid_loss)

    table["text"] = eval_batch["text"]
    input_tensors = [torch.tensor(t).long().to(device) for t in eval_batch["input_ids"]]

    model.eval()

    response_tensors_ref, response_tensors = [], []
    for i in range(len(input_tensors)):
        query_len = np.sum(eval_batch["attention_mask"][i])
        input_ids = input_tensors[i][
            max(query_len - config["input_size"], 0) : query_len
        ]

        output_ref = model_ref.generate(
            input_ids.unsqueeze(dim=0).to(device),
            max_length=len(input_ids) + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors_ref.append(clip_response(output_ref, len(input_ids)))

        output = model.generate(
            input_ids.unsqueeze(dim=0).to(device),
            max_length=len(input_ids) + config["output_size"],
            **gen_kwargs,
        ).squeeze()
        response_tensors.append(clip_response(output, len(input_ids)))

    table["original_model_response"] = [
        tokenizer.decode(r) for r in response_tensors_ref
    ]
    table["distil_model_response"] = [tokenizer.decode(r) for r in response_tensors]

    return logs, table


def clip_response(response, query_len):
    response = response[query_len:]
    stop_idx = (response == torch.tensor(198)).nonzero().flatten()
    if len(stop_idx) > 0:
        response = response[: stop_idx[0] + 1]
    return response


if __name__ == "__main__":
    model_path = "/tmp/distil_model/checkpoint_5/model.pt"
    checkpoint = torch.load(model_path)

    old_logs, old_table = evaluation()
    print("Original model loss: {}".format(old_logs["loss/validation"]))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    logs, table = evaluation()
    print("Distilled model loss: {}".format(logs["loss/validation"]))
