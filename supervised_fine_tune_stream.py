
# Some code based on https://github.com/epfml/landmark-attention
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

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from multiprocessing import cpu_count
from datasets import load_dataset
from tqdm import tqdm
import psutil

import torch
import transformers
# from torch.utils.data import Dataset, IterableDataset
from datasets.iterable_dataset import IterableDataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_replace import replace_llama_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root,f)
            yield fullname



PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input_seg}\n\n### Question:\n{question}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_size: int = field(default=None, metadata={"help": "for calculate max steps."})
    gpu_size: int = field(default=None, metadata={"help": "for calculate max steps and for logging for calcuated intervel."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



tok_example_count = 0

def preprocess_data(data_path, tokenizer):

    def _tokenize_fn(text: str) -> Dict:
        """Tokenize a list of strings."""
        tokenized = tokenizer(
                text,
                return_tensors="pt",
                # padding="longest",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

        input_ids = labels = tokenized.input_ids[0]
        input_ids_lens = labels_lens = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )
    
    def _process_function(example):

        global tok_example_count
        tok_example_count += 1
        if tok_example_count % 128 == 0:
            logging.warning(f"tok_example_count: {tok_example_count}")
   
        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        source = prompt_input.format_map(example) if example.get("input_seg", "") != "" else prompt_no_input.format_map(example)
            
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        target = f"{example['output']}{DEFAULT_EOS_TOKEN}"

        source_target = source + target

        example_tokenized = _tokenize_fn(source_target)
        source_tokenized = _tokenize_fn(source)

        input_ids = example_tokenized["input_ids"]
        label = copy.deepcopy(input_ids)
        label[:source_tokenized["input_ids_lens"]] = IGNORE_INDEX

        new_example = {"input_ids": input_ids, "labels": label}
   
        return new_example

    base = data_path
    sub_data_path_list = []
    for sub_data_path in findAllFile(base):
        sub_data_path_list.append(sub_data_path)

    logging.warning(f"before loading data, RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    list_data_dict = load_dataset('json', data_files=sub_data_path_list, split = f'train', streaming=True)
    logging.warning(f"list_data_dict: {list_data_dict}")
    logging.warning(f"after loading data, RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    logging.warning("Tokenizing inputs... This may take some time...")
    tokenized_dataset = list_data_dict.map(_process_function, remove_columns=["instruction", "input_seg", "question", "output"])

    return tokenized_dataset
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # logging.warning(f"instances: {instances}")
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # logging.warning(f"input_ids: {input_ids}")
        # logging.warning(f"labels: {labels}")
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = preprocess_data(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # logging.warning(f"train_dataset: {train_dataset}")
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    replace_llama_attn(training_args.use_flash_attn, True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if training_args.low_rank_training:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    logging.warning(f"data_module: {data_module}")
    # data_size = 2636762
    # data_size = 7325
    # GPU_size = 8
    training_args.max_steps = math.ceil(training_args.num_train_epochs * data_args.data_size/(training_args.per_device_train_batch_size * data_args.gpu_size))
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
