<h1 align="center">TableLlama <br> Towards Open Large Generalist Models for Tables</h1>

<div align="center">
 🔥 🔥 🔥 This repo contains the code, data, and models for <a href="https://arxiv.org/pdf/2311.09206.pdf">TableLlama</a>.
Check out our <a href = "https://osu-nlp-group.github.io/TableLlama/">[Project Page]</a> for more results and analysis!
</div>

<br>
<div align="center">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/tablellama_figure1.png" width="100%" title="Introduction Figure">
</div>
Figure 1: An overview of TableInstruct and TableLlama. TableInstruct includes a wide variety of realistic tables and tasks with instructions. We make the first step towards developing open-source generalist models for tables with TableInstruct and TableLlama.

<br>
<div align="center">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/tablellama_figure2.png" width="100%" title="Examplars Figure">
</div>
Figure 2: Illustration of three exemplary tasks: (a) Column type annotation. This task is to annotate the selected column with the correct semantic types. (b) Row population. This task is to populate rows given table metadata and partial row entities. (c) Hierarchical table QA. For subfigures (a) and (b), we mark candidates with red color in the "task instruction" part. The candidate set size can be hundreds to thousands in TableInstruct.


<h3>Release progress</h3>

- :ballot_box_with_check: Training Dataset for TableLlama (check `/data_v3` of 🤗 [TableInstruct Dataset](https://huggingface.co/datasets/osunlp/TableInstruct/)) 
- :ballot_box_with_check: TableLlama-7B model 
- :ballot_box_with_check: Code for Fine-tuning and Inference 
- :ballot_box_with_check: Evaluate Dataset of TableInstruct (check `/eval_data` of 🤗 [TableInstruct Dataset](https://huggingface.co/datasets/osunlp/TableInstruct/)) 
<!-- - :white_large_square: Code for Fine-tuning and Centralized training (TODO) -->

<h3>Updates</h3>

- 2024/3/13: Our paper has been accepted by NAACL 2024!
- 2024/3/21: We refine the prompts of 4 out-of-domain evaluation datasets: FEVEROUS, HybridQA, WikiSQL and WikiTQ of [TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct/) and update the results. Check the new results!
- 2024/3/21: We add the results of closed-source LLMs: GPT-3.5 and GPT-4.

### Datasets and Models
Our dataset and models are all available at Huggingface.

🤗 [TableInstruct Dataset](https://huggingface.co/datasets/osunlp/TableInstruct/)
                                       	
🤗 [TableLlama-7B](https://osu-nlp-group.github.io/TableLlama/)   	

The model is fine-tuned with the TableInstruct dataset using LongLoRA (7B), fully fine-tuning version as the base model, which replaces the vanilla attention mechanism of the original Llama-2 (7B) with shift short attention. The training takes 9 days on a 48 80*A100 cluster. Check out our paper for more details.

TableInstruct includes a comprehensive table-based instruction tuning dataset that covers a variety of real-world tables and realistic tasks. We include 14 datasets of 11 tasks in total. 

The model is evaluated on 8 in-domain datasets of 8 tasks and 6 out-of-domain datasets of 4 tasks.


## **Introduction**
We introduce TableLlama and TableInstruct: the FIRST open-source generalist LLM and instruction tuning dataset for tables. The TableLlama model is trained on TableInstruct Dataset, a meticulously curated instruction tuning dataset for tables. TableLlama is tuned on **2.6 million** table-based task data, and can handle up to **8K** context!


## **Installation**

Clone this repository and install the required packages:

```bash
git clone https://github.com/OSU-NLP-Group/TableLlama.git
cd TableLlama
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## **Training and Inference**

### **Fine-tuning**

To train the 7B model, run:

```bash
torchrun --nproc_per_node=8 supervised_fine_tune.py  \
        --model_name_or_path $MODEL_DIR \
        --bf16 True \
        --output_dir $OUTPUT_DIR  \
        --model_max_length 8192 \
        --use_flash_attn True \
        --data_path $DATA_DIR \
        --cache_dir /ML-A800/hf_cache  \
        --low_rank_training False \
        --num_train_epochs 2  \
        --per_device_train_batch_size 3     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 1     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 2000     \
        --save_total_limit 4     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_ratio 0.03     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 1     \
        --deepspeed "/ds_configs/stage2.json" \
        --tf32 True \
        --run_name $RUN_NAME
```

**Addressing OOM**
To train the 7B model with super large data size, if you encounter OOM issue, we provide code for streaming. You can run:
```bash
torchrun --nproc_per_node=8 supervised_fine_tune_stream.py  \
        --model_name_or_path $MODEL_DIR \
        --bf16 True \
        --output_dir $OUTPUT_DIR  \
        --model_max_length 8192 \
        --use_flash_attn True \
        --data_path $DATA_DIR \
        --gpu_size $GPU_SIZE \
        --data_size $DATA_SIZE \
        --cache_dir /ML-A800/hf_cache  \
        --low_rank_training False \
        --num_train_epochs 2  \
        --per_device_train_batch_size 3     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 1     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 2000     \
        --save_total_limit 4     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_ratio 0.03     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 1     \
        --deepspeed "/ds_configs/stage2.json" \
        --tf32 True \
        --run_name $RUN_NAME
       
```

### **Inference**
```bash
python3 inference_rel_extraction_col_type.py  \
        --base_model $MODEL_DIR \
        --context_size 8192 \
        --max_gen_len 128 \
        --flash_attn True \
        --input_data_file  /test_data/test_col_type.json \
        --output_data_file $OUTPUT_DIR/col_type_pred.json
```

## **Evaluation**

The folder `eval_scripts` includes evaluation scripts for all the in-domain test sets. To run the script, take HiTab (hierarchical table QA task) as an example:

```bash
cd eval_scripts
python evaluate_hitab.py --file_pred $OUTPUT_DIR/hitab_pred.json
```

## Prompt Format

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Question:
{question}

### Response:
```


- The instruction is designed to point out the task and give a detailed task description.
- The input is designed to provide the information about the table. We concatenate table metadata (if any) such as the Wikipedia page title, section
title and table caption with the serialized table as table input. We use '[TLE]' to represent the beginning of the table metadata, and use '[TAB]' to represent the beginning of the serialized table.
- The question is to accommodate all the information the model needs to complete the task and prompt the model to generate an answer.
- Task prompts examples (For more example prompts for other tasks, please refer to Appendix E in our paper.)

<br>
<div align="center">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/hitab.png" width="100%" title="hierarchical table qa">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/fetaqa.png" width="100%" title="hierarchical table qa">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/hybridqa.png" width="100%" title="hierarchical table qa">
  <img src="https://github.com/OSU-NLP-Group/TableLlama/blob/main/imgs/tabfact.png" width="100%" title="hierarchical table qa">
</div>

**Note:** 

- If you directly use our model for inference on your data, please make sure you organize the data in the same way as the examples shown above and in our paper Appendix. The performance will vary significantly along with the prompts.


## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@misc{zhang2023tablellama,
  title={TableLlama: Towards Open Large Generalist Models for Tables}, 
  author={Tianshu Zhang and Xiang Yue and Yifei Li and Huan Sun},
  year={2023},
  eprint={2311.09206},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```


