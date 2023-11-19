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
from torch.autograd import Variable
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
from src.gpt2_rl.flops_compute import TransformerHparams
from src.gpt2_rl.controller import Controller

from src.gpt2.configuration_gpt2 import GPT2Config
from src.gpt2.modeling_gpt2 import GPT2LMHeadModel

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
    trainable: Optional[str] = field(default="q_proj,v_proj")
    num_train_epochs: Optional[int] = field(default=2)

    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.2)
    lora_alpha: Optional[float] = field(default=32.)
    adapter_rank: Optional[int] = field(default=8)
    adapter_dropout: Optional[float] = field(default=0.2)

    modules_to_save: Optional[str] = field(default='embed_tokens,lm_head')
    debug_mode: Optional[bool] = field(default=False)
    peft_path: Optional[str] = field(default=None)
    leraning_rate: Optional[float] = field(default=1e-5)

    predict_with_generate: Optional[bool] = field(default=False)
    do_generation: Optional[bool] = field(default=False)

    do_train: Optional[bool] = field(default=True)
    use_consistency_loss: Optional[bool] = field(default=False)
    eval_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = field(default=5e-5)

    efficiency_coef: Optional[float] = field(default=2.0)
    entropy_coeff: Optional[float] = field(default=0.2)
    ema_baseline_decay: Optional[float] = field(default=0.92)
    use_return_dict: Optional[bool] = field(default=True)

    # search_space : Optional[str] = field(default="micro")
    #
    # # training_args.start_search_steps and completed_steps % training_args.search_every == 0 and completed_steps <= training_args.end_search_steps
    # start_search_steps: Optional[int] = field(default=500)
    # search_every: Optional[int] = field(default=200)
    # end_search_steps: Optional[int] = field(default=1200)


logger = logging.getLogger(__name__)


def eval_rl_model(model, controller, eval_dataloader, config):
    controller.eval()
    losses = []
    total_loss = 0.0
    num_batches = 0
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            # TODO: 第一次，完整的前向传播，对query进行表征，用于controller进行决策

            # controller进行预测, 进行action的选择

            # 再一次前向传播，有layer skipping， 计算target的loss

            input_key = ['input_ids', 'attenion_mask', 'labels']
            batch1 = {k: batch[k] for k in input_key if k in batch}
            batch1["layer_attn_skips"] = [0] * (config.n_layer)
            batch1["layer_ffn_skips"] = [0] * (config.n_layer)

            outputs = model(**batch1)
            hidden_states = outputs.hidden_states
            bsz = hidden_states.shape[0]
            sample_res, select_loss = controller.sample(hidden_states[:, -1, :])
            batch2 = {}
            batch2['input_ids'] = batch['input_ids2']
            batch2['attention_mask'] = batch['attention_mask2']
            batch2['labels'] = batch['labels2']
            sample_res = (torch.sum(sample_res.int(), 0) > int(0.5 * bsz)).int().tolist()
            batch2["layer_attn_skips"] = sample_res[0::2]
            batch2["layer_ffn_skips"] = sample_res[1::2]

            outputs2 = model(**batch2)
            cross_entropy = outputs2.loss
            skiped_layers = sum(sample_res)
            rewards = []
            for bs in range(bsz):
                rewards.append(-cross_entropy[bs].item() + skiped_layers)
            rs = np.array(rewards)
            baseline = np.mean(rs)
            policy_loss = 0
            for i in range(bsz):
                r = rewards[i] - baseline
                policy_loss += r * torch.mean(select_loss[i])
            loss = policy_loss

        total_loss += loss.item()
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
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)], )

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
        "use_cache": False
    }
    config = GPT2Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    # config.use_return_dict = training_args.use_return_dict
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

        query = example["query"]
        response = example["response"]

        input_ids = []
        labels = []
        input_ids2 = []
        labels2 = []

        input_1 = f"{query}"
        input_2 = f"{response}"
        input_ids_1 = tokenizer(input_1, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]
        input_ids_2 = tokenizer(input_2, return_tensors="pt")["input_ids"].cpu().numpy().tolist()[0]

        # 处理过长的样本
        target_length = min(len(input_ids_2), block_size - 128)
        query_length = block_size - target_length
        # print("target_length: ", target_length)
        # print("query_length: ", query_length)

        input_ids.extend(input_ids_1[-query_length:])
        labels.extend([-100] * len(input_ids_1[-query_length:]))

        input_ids2.extend(input_ids_1[-query_length:] + input_ids_2[: target_length])
        labels2.extend([-100] * len(input_ids_1[-query_length:]) + input_ids_2[: target_length])

        attention_mask = [1] * len(input_ids)
        attention_mask2 = [1] * len(input_ids2)

        assert len(input_ids) == len(labels) == len(attention_mask)
        assert len(input_ids2) == len(labels2) == len(attention_mask2)
        # print("length of input_ids: ", len(input_ids))
        # print("length of input_ids2: ", len(input_ids2))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_ids2": input_ids2,
            "attention_mask2": attention_mask2,
            "labels2": labels2,
            "target_length": target_length,
        }

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

    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples["input_ids"])
    #
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size + 1) * block_size
    #     # Split by chunks of max_len.
    #     result = {}
    #     for k, t in concatenated_examples.items():
    #         if total_length > len(t):
    #             if "input_ids" in k:
    #                 t = t + [tokenizer.eos_token_id] * (total_length - len(t))
    #             elif "attention_mask" in k:
    #                 t = t + [0] * (total_length - len(t))
    #             else:
    #                 t = t + [-100] * (total_length - len(t))
    #
    #         truncs = [t[i: i + block_size] for i in range(0, total_length, block_size)]
    #         result[k] = truncs
    #
    #     return result

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

    # tokenized_dataset = tokenized_dataset.map(
    #     group_texts,
    #     batched=True,
    #     # batch_size=1024,
    #     num_proc=8,
    #     load_from_cache_file=True,
    #     keep_in_memory=False,
    #     cache_file_names={k: os.path.join(data_args.dataset_name, f'grouped_{k}.arrow') for k in tokenized_dataset},
    #     desc=f"Grouping texts in chunks of {block_size}",
    # )
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

    controller = Controller(
        config.n_embd, config.num_hidden_layers
    ).to("cuda").to(torch.bfloat16)

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

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                       and (("lora" in n)
                            or ("adapter" in n) or ("gate" in n)
                            )],
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

    optimizer_grouped_parameters1 = [
        {
            "params": [p for n, p in controller.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    controller_optimizer = torch.optim.AdamW(optimizer_grouped_parameters1, lr=training_args.learning_rate)

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
    controller_lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=controller_optimizer,
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
    model, controller, optimizer, controller_optimizer, train_dataloader, eval_dataloader, lr_scheduler, controller_lr_scheduler = accelerator.prepare(
        model, controller, optimizer, controller_optimizer, train_dataloader, eval_dataloader, lr_scheduler, controller_lr_scheduler
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
        # total_model_params = 0
        # num_trained_params = 0
        for n, p in model.named_parameters():
            p.requires_grad = False
            # if ("lora" in n) or ("adapter" in n) or ("gate" in n):
            #     p.requires_grad = True
            # else:
            #     p.requires_grad = False
            # if p.requires_grad:
            #     num_trained_params += p.numel()
            # else:
            #     total_model_params += p.numel()
            # print(n, p.requires_grad)
        #
        # logger.info("Total Model Parameters: {}, "
        #             "Trainable Parameters: {}".format(
        #     total_model_params, num_trained_params))

        # training loop
        best_loss = 1000000000000
        best_loss_full_model = 1000000000000
        best_steps = None
        best_steps_full_model = None
        patience = 15

        # for REINFORCE
        baseline = None
        reward_history = []
        adv_history = []
        model.eval()
        for epoch in range(starting_epoch, training_args.num_train_epochs):

            total_loss = 0
            active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                # model.train()
                controller.train()
                # query 先过一遍前向传播，得到hidden_states
                batch_cur = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }
                batch_cur["layer_attn_gates"] = [1] * config.num_hidden_layers
                batch_cur["layer_ffn_gates"] = [1] * config.num_hidden_layers
                batch_cur["return_dict"] = True

                with torch.no_grad():
                    out = model(**batch_cur)
                    query_hidden_states = out.hidden_states

                # print("query_hidden_states: ", query_hidden_states.shape)

                # 计算reward：
                #    (1) 根据选择的actions，构造 layer_attn_gates， layer_ffn_gates
                #    (2) 计算样本在有skipping下的损失; 节省的复杂度
                #    (3) reward 组成： LM没有skipping的损失: A1; 在sample的action下面的skipping后损失函数: A2； 选择保留的层的复杂度: B
                #                      - (A1-A2) + (-B)

                # 没有skipping情况下的损失计算
                batch_cur = {
                    "input_ids": batch["input_ids2"],
                    "attention_mask": batch["attention_mask2"],
                    "labels": batch["labels2"],
                }
                batch_cur["layer_attn_gates"] = [1] * config.num_hidden_layers
                batch_cur["layer_ffn_gates"] = [1] * config.num_hidden_layers
                batch_cur["return_dict"] = True

                with torch.no_grad():
                    out = model(**batch_cur)
                    loss_no_skipping = out.loss
                # print("loss_no_skipping: ", loss_no_skipping)

                # 可以采样多次轨迹
                list_slipping_losses = []
                list_actions = []
                list_rewards = []

                controller_optimizer.zero_grad()
                for i in range(1):
                    actions, selected_log_probs, controller_entropies = controller.sample(
                        query_hidden_states
                    )
                    layer_attn_gates = actions[0::2]
                    layer_ffn_gates = actions[1::2]


                    batch_cur = {
                        "input_ids": batch["input_ids2"],
                        "attention_mask": batch["attention_mask2"],
                        "labels": batch["labels2"],
                    }
                    batch_cur["layer_attn_gates"] = layer_attn_gates
                    batch_cur["layer_ffn_gates"] = layer_ffn_gates
                    batch_cur["return_dict"] = True

                    with torch.no_grad():
                        out = model(**batch_cur)
                        loss_skipping_1 = out.loss
                        list_slipping_losses.append(loss_skipping_1)

                    # 计算reward
                    loss_delta = - (loss_skipping_1 - loss_no_skipping)

                    gpt2_calc = TransformerHparams(
                        config.hidden_size, config.num_hidden_layers,
                        s=128, v=config.vocab_size, output_frac=1.0
                    )
                    attn_flops = gpt2_calc.get_attn_flops()
                    ffn_flops = gpt2_calc.get_ffn_flops()
                    complexity_saved_attn = (config.num_hidden_layers - sum(layer_attn_gates)) * attn_flops
                    complexity_saved_ffn = (config.num_hidden_layers - sum(layer_ffn_gates)) * ffn_flops
                    complexity_saved = complexity_saved_attn + complexity_saved_ffn
                    complexity_whole = config.num_hidden_layers * (attn_flops + ffn_flops)
                    # print("complexity_saved: ", complexity_saved)
                    # print("complexity_whole: ", complexity_whole)

                    reward_ = loss_delta + training_args.efficiency_coef * complexity_saved / complexity_whole

                    reward_history.append(reward_)

                    # moving average baseline
                    if baseline is None:
                        baseline = reward_
                    else:
                        decay = training_args.ema_baseline_decay
                        baseline = decay * baseline + (1 - decay) * reward_

                    adv = reward_ - baseline
                    adv_history.append(adv)

                    # policy loss
                    loss = - selected_log_probs * Variable(torch.Tensor(adv).cuda())
                    loss = loss.mean()  # or loss.mean()
                    # loss -= training_args.entropy_coeff * controller_entropies.mean()

                    if random.uniform(0, 1) < 0.05:
                        print("actions: ", actions)
                        print("controller_entropies: ", controller_entropies)
                        print("controller_entropies mean: ", controller_entropies.mean())
                        print("layer_attn_gates: ", layer_attn_gates)
                        print("layer_ffn_gates: ", layer_ffn_gates)
                        print("loss_delta: ", loss_delta)
                        print("complexity_saved / complexity_whole: ", complexity_saved / complexity_whole)
                        print("reward_: ", reward_)
                        print("baseline: ", baseline)
                        print("adv: ", adv)
                        print("loss: ", loss)

                    loss.backward()

                torch.nn.utils.clip_grad_norm(controller.parameters(), 1.0)
                controller_optimizer.step()
                controller_lr_scheduler.step()

                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % training_args.eval_steps == 0 and completed_steps > 0:
                    eval_loss = eval_rl_model(
                        model,
                        controller,
                        eval_dataloader,
                        config
                    )

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_steps = completed_steps
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(controller)
                        unwrapped_model.save_pretrained(
                            training_args.output_dir, is_main_process=accelerator.is_main_process,
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
            response = outputs[0][input_length:]
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

    """
    # debug
    
    CUDA_VISIBLE_DEVICES="0" python -u src/gpt2_rl/run_reinforce_IWL.py --seed 600 --dataset_name datasets/ultraChat/flat_format --model_name_or_path ./resources/gpt2 --block_size 1024 --lora_rank 64 --adapter_rank 64 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 10 --warmup_steps 100 --output_dir experiments/iwl_gpt2_debug_0 --do_train --do_eval --eval_steps 50 --learning_rate 2e-4 --use_consistency_loss True --overwrite_output_dir 
    
    
    # gpt2-large
    CUDA_VISIBLE_DEVICES="0" nohup python -u src/gpt2_rl/run_reinforce_IWL.py --seed 600 --dataset_name datasets/ultraChat/flat_format --model_name_or_path ./experiments/gpt2_debug_0 --block_size 1024 --lora_rank 64 --adapter_rank 64 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --num_train_epochs 10 --warmup_steps 1000 --output_dir experiments/iwl_gpt2_debug_0 --do_train --do_eval --eval_steps 100000 --learning_rate 2e-4 --overwrite_output_dir > iwl_gpt2_debug_0.log &
    
    """
