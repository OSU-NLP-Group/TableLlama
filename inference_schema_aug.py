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
    # parser.add_argument('--question', type=str, default="")
    # parser.add_argument('--material', type=str, default="")
    # parser.add_argument('--material_title', type=str, default="")
    # parser.add_argument('--material_type', type=str, default="material")
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

# def format_prompt(material, message, material_type="book", material_title=""):
#     if material_type == "paper":
#         prompt = f"Below is a paper. Memorize the material and answer my question after the paper.\n {material} \n "
#     elif material_type == "book":
#         material_title = ", %s"%material_title if len(material_title)>0 else ""
#         prompt = f"Below is some paragraphs in the book{material_title}. Memorize the content and answer my question after the book.\n {material} \n "
#     else:
#         prompt = f"Below is a material. Memorize the material and answer my question after the material. \n {material} \n "
#     message = str(message).strip()
#     prompt += f"Now the material ends. {message}"

#     return prompt

# def read_txt_file(material_txt):
#     if not material_txt.split(".")[-1]=='txt':
#         raise ValueError("Only support txt or pdf file.")
#     content = ""
#     with open(material_txt) as f:
#         for line in f.readlines():
#             content += line
#     return content

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
        new_item["table_id"] = test_data[i]["table_id"]
        new_item["instruction"] = test_data[i]["instruction"]
        new_item["input_seg"] = test_data[i]["input_seg"]
        new_item["question"] = test_data[i]["question"]
        new_item["target"] = test_data[i]["target"]
        new_item["output_list"] = test_data[i]["output_list"]
        new_item["output"] = test_data[i]["output"]
        new_item["predict"] = output

        test_data_pred.append(new_item)
        # import pdb
        # pdb.set_trace() 
    with open(args.output_data_file, "w") as f:
        json.dump(test_data_pred, f, indent = 2)

    # output = respond(args.material, args.question, args.material_type, args.material_title)
    # print("output", output)

if __name__ == "__main__":
    args = parse_config()
    main(args)


# from dataclasses import dataclass, field

# import numpy as np
# import torch
# import transformers
# from transformers import GenerationConfig

# from train_llama2_long_context_reformat import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
#   DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT

# import json
# from tqdm import tqdm
# import math
# import argparse

# @dataclass
# class InferenceArguments:
#   model_max_length: int = field(
#     # default=512,
#     # default=1024,
#     default=1536,
#     metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
#   )
#   load_in_8bit: bool = field(
#     default=False,
#     metadata={"help": "Load the model in 8-bit mode."},
#   )
#   inference_dtype: torch.dtype = field(
#     default=torch.float32,
#     metadata={"help": "The dtype to use for inference."},
#   )
#   max_new_tokens: int = field(
#     default=64,
#     metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
#   )

# @dataclass
# class FileArguments:
#   input_data_file: str = field(
#     default="",
#     metadata={"help": ""},
#   )
#   output_data_file: str = field(
#     default="",
#     metadata={"help": ""},
#   )




# def batch_process(data_list, model, tokenizer, generation_config, batch_size, max_new_tokens):
#   pred = []
#   for i in tqdm(range(math.ceil(len(data_list)/batch_size))):
#     if i != math.ceil(len(data_list)/batch_size) - 1:
#       batch_data = data_list[i * batch_size: i * batch_size + batch_size]
#     else:
#       batch_data = data_list[i * batch_size:]
#     batch_prompt =[generate_prompt(item["instruction"], item["input_seg"], item["question"]) for item in batch_data]
#     inputs = tokenizer(batch_prompt, 
#                       return_tensors="pt", 
#                       padding="longest",
#                       max_length=tokenizer.model_max_length,
#                       truncation=True)
#     outputs = model.generate(input_ids=inputs["input_ids"].cuda(), generation_config=generation_config, max_new_tokens = max_new_tokens)
   
#     # import pdb
#     # pdb.set_trace()
#     # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
#     # generated_tokens = outputs.sequences[:, input_length:]
#     # pred += tokenizer.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
#     pred += tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
#     # import pdb
#     # pdb.set_trace()
#   return pred


# def inference(test_data, model_args, inference_args):
#   # parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
#   # model_args, inference_args = parser.parse_args_into_dataclasses()

#   model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     load_in_8bit=inference_args.load_in_8bit,
#     torch_dtype=inference_args.inference_dtype,
#     device_map="auto",
#   )
#   model.cuda()
#   model.eval()

#   generation_config = GenerationConfig(
#     temperature=0.1,
#     top_p=0.75,
#     # num_beams=4,
#     num_beams=1,
#     # num_beams=2,
#   )

#   tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_args.model_name_or_path,
#     use_fast=False,
#     model_max_length=inference_args.model_max_length,
#     padding_side="left"     ### important to add this in inference
#   )

#   if tokenizer.pad_token is None:
#     smart_tokenizer_and_embedding_resize(
#       special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
#       tokenizer=tokenizer,
#       model=model,
#     )
#   tokenizer.add_special_tokens(
#     {
#       "eos_token": DEFAULT_EOS_TOKEN,
#       "bos_token": DEFAULT_BOS_TOKEN,
#       "unk_token": DEFAULT_UNK_TOKEN,
#     }
#   )

#   pred = batch_process(test_data, model, tokenizer, generation_config, 1, inference_args.max_new_tokens)
 
#   new_test_list = []
#   for i in tqdm(range(len(test_data))):
#   # for i in tqdm(range(90, 101)):
#   # for i in tqdm(range(3)):
#       instruction = test_data[i]["instruction"]
#       item = {}
#       item["idx"] = i
#       # item["table_id"] = test_data[i]["table_id"]
#       # item["entity"] = test_data[i]["entity"]
#       item["instruction"] = instruction
#       # item["input"] = input
#       item["input_seg"] = test_data[i]["input_seg"]
#       # item["tokenizer_tensor_shape"] = inputs["input_ids"].shape
#       item["output"] = test_data[i]["output"]
#       item["predict"] = pred[i]

#       new_test_list.append(item)
#   return new_test_list


# if __name__ == "__main__":

#   parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments, FileArguments))
#   model_args, inference_args, file_args = parser.parse_args_into_dataclasses()

#   # num = 0
#   # with open("/users/PAA0201/shubaobao/stanford_alpaca/table_all_tasks_fair/test/split_16_col_type/test_" + str(file_args.input_data_file_num) + ".json", "r") as f:
#   with open(file_args.input_data_file, "r") as f:
#       test_data = json.load(f)

#   # import random
#   # test_data = random.sample(test_data, k=5)
#   test_list = inference(test_data, model_args, inference_args)
  
#   # with open("/users/PAA0201/shubaobao/stanford_alpaca/table_all_tasks_fair/pred_ser_20000_seg/test_beam_search/test_" + str(file_args.output_data_file_num) + ".json", "w") as f:
#   with open(file_args.output_data_file, "w") as f:
#     json.dump(test_list, f, indent = 2)
  
#   print("input file:", str(file_args.input_data_file))
#   print("output file:", str(file_args.output_data_file))
