import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import pandas as pd
from utils import Trainer, add_labels, get_gpu, load_data_from_uri, set_training_arguments, tokenize_function

warnings.filterwarnings("ignore")

# Finetuning EleutherAI model
def get_model_from_huggingface(uri:str):
    model = None
    tokenizer = None
    if uri.find("EleutherAI")<0:
        raise ValueError("Only Eleuther AI model is supported")
    else:
        model = AutoModelForCausalLM.from_pretrained(uri)
        tokenizer = AutoTokenizer.from_pretrained(uri)
    
    return model, tokenizer

def preprocess_data(uri:str, tokenizer: AutoTokenizer):
    data = load_data_from_uri(uri)
    # tokenize_func = tokenize_function(tokenizer=tokenizer)
    tokenized_data = data.map(
    tokenize_function(tokenizer),
    batched=True,
    batch_size=1,
    drop_last_batch=True
    )
    import pdb
    pdb.set_trace()
    # tokenized_data.map(add_labels)
    split_dataset = tokenized_data["train"].train_test_split(test_size=0.1, shuffle=True, seed=123)
    return split_dataset

if __name__ == "__main__":
    model, tokenizer = get_model_from_huggingface("EleutherAI/pythia-70m")
    dev = get_gpu()
    model.to(dev)
    dataset = preprocess_data("BashitAli/Indian_history", tokenizer=tokenizer)
    print(dataset)
    max_steps = 10
    ta = set_training_arguments(max_steps=max_steps)
    model_flops = (
    model.floating_point_ops(
        {
        "input_ids": torch.zeros(
            (1, 2048)
        )
        }
    )
    * ta.gradient_accumulation_steps
    )
    trainer = Trainer(model, model_flops=model_flops, total_steps=max_steps, args=ta, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()
    
