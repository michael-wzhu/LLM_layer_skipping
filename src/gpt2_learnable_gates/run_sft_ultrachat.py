#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed, DataCollatorForSeq2Seq, GenerationConfig, get_scheduler, AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

sys.path.append("./")

from src.gpt2.configuration_gpt2 import GPT2Config
from src.gpt2_learnable_gates.modeling_gpt2 import GPT2LMHeadModel

os.environ["WANDB_MODE"] = "disabled"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "where to store the cached data."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    num_train_epochs : Optional[int] = field(default=2)

    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.2)
    lora_alpha : Optional[float] = field(default=32.)
    adapter_rank: Optional[int] = field(default=8)
    adapter_dropout: Optional[float] = field(default=0.2)

    modules_to_save : Optional[str] = field(default='embed_tokens,lm_head')
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    leraning_rate : Optional[float] = field(default=1e-5)

    predict_with_generate : Optional[bool] = field(default=False)
    do_generation : Optional[bool] = field(default=False)

    do_train: Optional[bool] = field(default=True)
    use_consistency_loss: Optional[bool] = field(default=False)
    eval_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = field(default=5e-5)

    # search_space : Optional[str] = field(default="micro")
    #
    #
    # # training_args.start_search_steps and completed_steps % training_args.search_every == 0 and completed_steps <= training_args.end_search_steps
    # start_search_steps: Optional[int] = field(default=500)
    # search_every: Optional[int] = field(default=200)
    # end_search_steps: Optional[int] = field(default=1200)


logger = logging.getLogger(__name__)


def eval_model(model, eval_dataloader, layer_attn_masks=None, layer_ffn_masks=None):
    model.eval()
    losses = []
    total_loss = 0.0
    num_batches = 0
    for step, batch in tqdm(enumerate(eval_dataloader)):
        batch["layer_attn_masks"] = layer_attn_masks
        batch["layer_ffn_masks"] = layer_ffn_masks
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss
        num_batches += 1

    try:
        eval_loss = total_loss / num_batches
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
        eval_loss = 1000000000

    return eval_loss


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = GPT2Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    config.lora_rank = training_args.lora_rank
    config.lora_dropout = training_args.lora_dropout
    config.adapter_rank = training_args.adapter_rank
    config.adapter_dropout = training_args.adapter_dropout

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(example):

        # max_seq_length = 1024

        dialogues = example.get("data")
        # print(len(dialogues))
        assert len(dialogues) % 2 == 0

        input_ids = []
        labels = []
        for round_idx in range(int(len(dialogues) / 2)):
            query_ = dialogues[2 * round_idx]
            response_ = dialogues[2 * round_idx + 1]

            input_1 = f"<|endoftext|>user:\n{query_}\n"
            input_2 = f"<|endoftext|>assistant:\n{response_}<|endoftext|>"
            input_ids_1 = tokenizer(input_1, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]
            input_ids_2 = tokenizer(input_2, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]

            input_ids.extend(input_ids_1 + input_ids_2)
            labels.extend(input_ids_1 + input_ids_2)

        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(labels) == len(attention_mask)
        # print("length of input_ids: ", len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # def tokenize_function_eval(example):
    #
    #     input = example.get("input")
    #     if not input:
    #         input = example.get("instruction")
    #
    #     eos_token_id = 50256
    #
    #     input_ids_1 = tokenizer(input, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]
    #     response_str = "<response>"
    #     response_str_ids_1 = tokenizer(response_str, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]
    #
    #     input_ids = [eos_token_id] + input_ids_1 + response_str_ids_1
    #     attention_mask = [1] * len(input_ids)
    #
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }



    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size + 1) * block_size
        # Split by chunks of max_len.
        result = {}
        for k, t in concatenated_examples.items():
            if total_length > len(t):
                if k == "input_ids":
                    t = t + [tokenizer.eos_token_id] * (total_length - len(t))
                elif k == "attention_mask":
                    t = t + [0] * (total_length - len(t))
                else:
                    t = t + [-100] * (total_length - len(t))

            truncs = [t[i : i + block_size] for i in range(0, total_length, block_size)]
            result[k] = truncs

        # for k in result:
        #     print(k, len(result[k]))
        return result

    # with training_args.main_process_first(desc="dataset map tokenization and grouping"):
    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.dataset_name, "train.json"),
            # "dev": os.path.join(data_args.dataset_name, "dev.json"),
            "test": os.path.join(data_args.dataset_name, "test.json"),
        },
        # cache_dir=data_args.dataset_cache_dir,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenized_dataset = raw_datasets.map(
                tokenize_function,
                batched=False,
                num_proc=8,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=True,
                cache_file_names={k: os.path.join(data_args.dataset_name, f'tokenized_{k}.arrow') for k in raw_datasets},
                desc="Running tokenizer on dataset",
            )
    print("tokenized_dataset: ", tokenized_dataset)
    print(tokenized_dataset["train"][3]['input_ids'])
    print(tokenized_dataset["train"][3]['labels'])

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        # batch_size=1024,
        num_proc=8,
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_names = {k: os.path.join(data_args.dataset_name, f'grouped_{k}.arrow') for k in tokenized_dataset},
        desc=f"Grouping texts in chunks of {block_size}",
    )
    lm_datasets = tokenized_dataset

    # lm_datasets = tokenized_dataset["train"].train_test_split(test_size=0.02)
    lm_datasets["dev"] = lm_datasets["test"]
    print(lm_datasets)

    # test_dataset = raw_datasets["dev"].map(
    #             tokenize_function_eval,
    #             batched=False,
    #             num_proc=1,
    #             desc="Running tokenizer on test dataset",
    #         )
    # print(test_dataset)

    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))

    if training_args.do_eval:
        eval_dataset = lm_datasets["dev"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    # torch_dtype = model_args.torch_dtype
    # config.num_hidden_layers = 2
    # model = QWenLMHeadModel._from_config(
    #     config
    # )
    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.bfloat16,
    )
    # model = model.to(torch.device("cuda"))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding='longest'
    )

    # Initialize our Trainer
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    eval_indices = list(range(len(eval_dataset)))
    random.shuffle(eval_indices)
    valid_dataset = eval_dataset.select(eval_indices[: 16])
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and (("lora" in n)
                            or ("adapter" in n) or ("gate" in n)
                            ) ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                       and (("lora" in n)
                            or ("adapter" in n) or ("gate" in n)
                            )],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_train_steps * training_args.gradient_accumulation_steps,
    )

    # accelerator
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["project_dir"] = training_args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        **accelerator_log_kwargs
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        os.makedirs(training_args.output_dir, exist_ok=True)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    if training_args.do_train:

        # Train!
        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Training stage 1:
        # 第一阶段：lora参数初步训练
        total_model_params = 0
        num_trained_params = 0
        for n, p in model.named_parameters():
            if ("lora" in n) or ("adapter" in n) or ("gate" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
            if p.requires_grad:
                num_trained_params += p.numel()
            else:
                total_model_params += p.numel()
            print(n, p.requires_grad)

        logger.info("Total Model Parameters: {}, "
                    "Trainable Parameters: {}".format(
            total_model_params, num_trained_params))

        # training loop
        best_loss = 1000000000000
        best_loss_full_model = 1000000000000
        best_steps = None
        best_steps_full_model = None
        patience = 15
        for epoch in range(starting_epoch, training_args.num_train_epochs):

            total_loss = 0
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):

                # layer_dropping_rate = 0.0
                if completed_steps < 500:
                    layer_dropping_rate = 0.0
                elif completed_steps < 1500:
                    layer_dropping_rate = 0.1
                elif completed_steps < 2500:
                    layer_dropping_rate = 0.2
                elif completed_steps < 3500:
                    layer_dropping_rate = 0.3
                elif completed_steps < 4500:
                    layer_dropping_rate = 0.4
                else:
                    layer_dropping_rate = 0.5

                model.train()
                with accelerator.accumulate(model):

                    if not training_args.use_consistency_loss:
                        batch["layer_attn_masks"] = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()
                        batch["layer_ffn_masks"] = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()

                        outputs = model(**batch)
                        loss = outputs.loss
                    else:
                        layer_attn_masks_1 = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()
                        layer_ffn_masks_1 = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()
                        layer_attn_masks_2 = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()
                        layer_ffn_masks_2 = np.random.binomial(
                            1,
                            1 - layer_dropping_rate,
                            config.num_hidden_layers,
                        ).tolist()

                        loss = model.consistency_loss(
                            layer_attn_masks_1,
                            layer_ffn_masks_1,
                            layer_attn_masks_2,
                            layer_ffn_masks_2,
                            **batch
                        )
                    total_loss += loss.detach().float()

                    if random.uniform(0, 1) < 0.01:
                        print("loss: ", loss.detach().float())
                        print("model.layer_attn_gates: ", model.layer_attn_gates.data.to(torch.float32).detach().cpu().numpy().tolist())
                        print("model.layer_ffn_gates: ", model.layer_ffn_gates.data.to(torch.float32).detach().cpu().numpy().tolist())

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if completed_steps % training_args.eval_steps == 0 and completed_steps > 0:
                        eval_loss = eval_model(
                            model,
                            eval_dataloader,
                            layer_attn_masks=[1, 1, 1, 1] * int(config.num_hidden_layers / 4),
                            layer_ffn_masks=[1, 1, 1, 1] * int(config.num_hidden_layers / 4),
                        )
                        if eval_loss < best_loss_full_model:
                            best_loss_full_model = eval_loss
                            best_steps_full_model = completed_steps
                        logger.info(f"current eval_loss with full model: {eval_loss}; completed_steps: {completed_steps}")
                        logger.info(f"current best_loss_full_model: {best_loss_full_model}; best_steps_full_model: {best_steps_full_model}")

                        # eval model with structural drop
                        eval_loss = eval_model(
                            model,
                            eval_dataloader,
                            layer_attn_masks=[1, 0, 1, 0] * int(config.num_hidden_layers / 4),
                            layer_ffn_masks=[0, 1, 0, 1] * int(config.num_hidden_layers / 4),
                        )
                        logger.info(f"completed_steps: {completed_steps}; eval loss: {eval_loss}")
                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_steps = completed_steps
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                training_args.output_dir, is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(training_args.output_dir)

                            # logger.info(f"best_loss: {best_loss}; best_steps: {best_steps}")
                        logger.info(f"current best_loss: {best_loss}; best_steps: {best_steps}")

            print("avg loss: ", total_loss.item() / len(train_dataloader))
            if completed_steps >= training_args.max_train_steps:
                break

        logger.info("*" * 50)
        logger.info(f"best steps: {best_steps}; best loss: {best_loss}")
        logger.info("*" * 50)

    if training_args.do_generation:
        model.eval()

        generation_config = GenerationConfig.from_dict(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": False,
                "top_k": 0,
                "top_p": 0.0,
                "num_beams": 5,
                "repetition_penalty": 1.05,
                "max_new_tokens": 5
            }
        )

        list_predicted_samples = []

        for samp in test_dataset:
            # print(samp)
            input_ids = [samp["input_ids"]]
            attention_mask = [samp["attention_mask"]]
            input_length = len(input_ids[0])

            outputs = model.generate(
                torch.LongTensor(input_ids).to(torch.device("cuda:0")),
                attention_mask=torch.LongTensor(attention_mask).to(torch.device("cuda:0")),
                generation_config=generation_config,
            )
            response = outputs[0][input_length: ]
            eod_token_idx = None

            for i in range(len(response)):
                if response[i] in [tokenizer.eos_token_id]:
                    eod_token_idx = i
                    break
            if eod_token_idx is None:
                eod_token_idx = len(response) - 1

            response = response[: eod_token_idx]
            response_text = tokenizer.decode(
                response
            )
            samp_copy = copy.deepcopy(samp)
            samp_copy["pred"] = response_text
            list_predicted_samples.append(
                samp_copy
            )
            with open(os.path.join(training_args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
                for samp in list_predicted_samples:
                    f.write(
                        json.dumps(samp, ensure_ascii=False) + "\n"
                    )


if __name__ == "__main__":
    main()

    '''
    # debug
    CUDA_VISIBLE_DEVICES="2" nohup python -u src/gpt2_learnable_gates/run_sft_ultrachat.py --seed 700 --dataset_name datasets/ultraChat/ --model_name_or_path resources/gpt2-large --block_size 640 --lora_rank 64 --adapter_rank 64 --per_device_train_batch_size 4 --per_device_eval_batch_size 12 --gradient_accumulation_steps 12 --num_train_epochs 10 --warmup_steps 100 --output_dir experiments/gpt2_debug_3 --do_train --do_eval --eval_steps 50 --learning_rate 2e-4 --use_consistency_loss True --overwrite_output_dir > train_3.log & 
    
    
    '''
