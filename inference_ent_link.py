import os
import json
import sys
import math
import torch
import argparse
# import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig
from llama_attn_replace import replace_llama_attn
from supervised_fine_tune import PROMPT_DICT
from tqdm import tqdm
# from queue import Queue
# from threading import Thread
# import gradio as gr

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--input_data_file', type=str, default='input_data/', help='')
    parser.add_argument('--output_data_file', type=str, default='output_data/', help='')
    args = parser.parse_args()
    return args

def generate_prompt(instruction, question, input_seg=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(item):
    # def response(material, question, material_type="", material_title=None):
        # material = read_txt_file(material)
        # prompt = format_prompt(material, question, material_type, material_title)
        prompt = generate_prompt(instruction = item["instruction"], input_seg = item["input_seg"], question = item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
        )
        out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        out = out.split(prompt)[1].strip()
        return out

    return response

def main(args):
    if args.flash_attn:
        replace_llama_attn()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        # padding_side="right",
        padding_side="left",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(args.input_data_file, "r") as f:
        test_data = json.load(f)
    
    # import random
    # test_data = random.sample(test_data, k=5)

    test_data_pred = []
    for i in tqdm(range(len(test_data))):
        item = test_data[i]
        new_item = {}
        respond = build_generator(item, model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=not args.flash_attn)   # the temperature and top_p are highly different with previous alpaca exp, pay attention to this if there is sth wrong later
        output = respond(item)

        new_item["idx"] = i
        new_item["table_id"] = test_data[i]["id"]
        new_item["instruction"] = test_data[i]["instruction"]
        new_item["input_seg"] = test_data[i]["input_seg"]
        new_item["question"] = test_data[i]["question"]
        new_item["candidates_list"] = test_data[i]["candidates_list"]
        new_item["candidates_entity_desc_list"] = test_data[i]["candidates_entity_desc_list"]
        new_item["output"] = test_data[i]["output"]
        new_item["predict"] = output

        test_data_pred.append(new_item)
        # import pdb
        # pdb.set_trace() 
    with open(args.output_data_file, "w") as f:
        json.dump(test_data_pred, f, indent = 2)

if __name__ == "__main__":
    args = parse_config()
    main(args)


