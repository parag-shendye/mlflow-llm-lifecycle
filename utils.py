"""Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
"""

import datetime
from typing import Union, Optional
import pandas as pd
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from transformers import TrainingArguments
import transformers
import time

load_dotenv(find_dotenv())
EXPERIMENT = "llm-baseline"
mlflow.set_experiment(EXPERIMENT)
mlflow.pytorch.autolog()


def load_data_from_uri(uri: Optional[str]):
    if isinstance(uri, str) and len(uri) != 0:
        try:
            data = load_dataset(uri)
            return data
        except:
            raise ValueError("Data could not be found")
    return None


def tokenize_function(tokenizer):
    def tokenize(examples):
        if "question" in examples and "answer" in examples:
            text = examples["question"][0] + examples["answer"][0]
        elif "input" in examples and "output" in examples:
            text = examples["input"][0] + examples["output"][0]
        elif "Question" in examples and "Answer" in examples:
            text = examples["Question"][0] + examples["Answer"][0]
        elif "instruction" in examples and "response" in examples:
            text = examples["instruction"][0] + examples["response"][0]
        else:
            text = examples["text"][0]

        tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
        )

        max_length = min(
            tokenized_inputs["input_ids"].shape[1],
            2048
        )
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]

        return tokenized_inputs
    return tokenize


def get_gpu():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        return torch.device("cuda")
    else:
        return torch.device('cpu')


def set_training_arguments(lr=1.0e-5, epochs=1, max_steps=10, output_dir="output"):
    return TrainingArguments(
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        output_dir=output_dir,
        overwrite_output_dir=False,
        disable_tqdm=False,
        eval_steps=120,
        save_steps=120,
        warmup_steps=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        optim="adafactor",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )


class Trainer(transformers.Trainer):
    def __init__(
        self,
        model,
        model_flops,
        total_steps,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
    ):
        super(Trainer, self).__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )

        self.total_steps = total_steps
        self.model_flops = model_flops
        self.start_step = 0

    def training_step(self, model, inputs):
        if inputs["input_ids"].numel() == 0:

            print("Inputs: ", inputs)
            print("Inputs - input_ids", inputs["input_ids"])
            print("numel", inputs["input_ids"].numel())

            return torch.tensor(0)
        else:
            model.train()
            inputs = self._prepare_inputs(inputs)
            print(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                self.accelerator.backward(loss)

            return loss.detach() / self.args.gradient_accumulation_steps

    def log(self, logs):
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.update_log_timing(logs)

        output = {**logs, **{"step": self.state.global_step}}
        self.update_history(output)

        # logger.debug("Step (" + str(self.state.global_step) + ") Logs: " + str(logs))
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def update_log_timing(self, logs):
        if len(self.state.log_history) == 0:
            self.start_time = time.time()
            logs["iter_time"] = 0.0
            logs["flops"] = 0.0
            logs["remaining_time"] = 0.0
            self.start_step = self.state.global_step
        elif self.state.global_step > self.start_step:
            logs["iter_time"] = (time.time() - self.start_time) / (
                self.state.global_step - self.start_step
            )
            logs["flops"] = self.model_flops / logs["iter_time"]
            logs["remaining_time"] = (self.total_steps - self.state.global_step) * logs[
                "iter_time"
            ]

    def update_history(self, output):
        if "eval_loss" in output:
            return
        if len(self.state.log_history) > 0:
            smoothing_window = 100
            p = 1.0 / smoothing_window
            if "loss" in output:
                output["loss"] = output["loss"] * p + self.state.log_history[-1][
                    "loss"
                ] * (1.0 - p)
        self.state.log_history.append(output)


def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )
    print(input_ids)
    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(
        generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer
