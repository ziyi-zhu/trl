import torch
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


config = {
    "project_name": str(os.environ.get("PROJECT_NAME", "gpt2-ppo")),
    "auth_token": "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",
    "model_name": str(os.environ.get("MODEL_NAME", "gpt2")),
    "tokenizer_name": str(os.environ.get("TOKENIZER_NAME", "gpt2")),
    "vf_model_name": str(os.environ.get("VF_MODEL_NAME", "gpt2")),
    "ref_model_name": str(os.environ.get("REF_MODEL_NAME", "gpt2")),
    "cls_model_name": str(
        os.environ.get("CLS_MODEL_NAME", "ChaiML/rewardModel90kEpoch2K1M3")
    ),
    "cls_tokenizer_name": str(
        os.environ.get("CLS_TOKENIZER_NAME", "roberta-large-mnli")
    ),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"], use_auth_token=config["auth_token"]
)


if __name__ == "__main__":
    load_path = "/tmp/checkpoint-90-state.pt"
    print("loading checkpoint from {}".format(load_path))

    checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    steps = checkpoint["steps"]

    model.eval()
    model.push_to_hub(
        "distil-gpt2-ppo-v1", organization="ChaiML", use_auth_token=config["auth_token"]
    )
