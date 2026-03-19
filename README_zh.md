# Midi-Tuning

[**Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue**](https://arxiv.org/abs/2402.06967) (ACL 2024)
（一次指令，多轮一致聊天：一种高效的对话微调框架）

我们提出了一个高效的\*\*多轮交互式对话微调（Midi-Tuning）**框架。该框架通过在大型语言模型（LLM）之上构建两个适配器（Adapters），分别对**智能体（Agent）**和**用户（User）**进行建模。适配器按轮次交替利用各自的话语，并通过**轮次级记忆缓存机制（round-level memory caching mechanism）\*\*进行微调。

\<p align="center"\>
\<img src="figure/overview.png" width="98%" /\>
\</p\>

## 环境要求

所需的软件包列在 `requirements.txt` 中。假设你使用 [Anaconda](https://www.anaconda.com/) 来管理 Python 依赖，可以通过运行以下命令进行安装：

```bash
conda create -n midi python=3.10
conda activate midi
pip install -r requirements.txt
```

## 数据集

我们在两个数据集上评估了 Midi-Tuning 框架：[LIGHT](https://aclanthology.org/D19-1062.pdf)（一个基于角色的对话数据集）和 [TopDial](https://aclanthology.org/2023.emnlp-main.72.pdf)（一个目标导向的主动对话数据集）。数据集可以从以下链接下载：

  - LIGHT ([GoogleDrive](https://drive.google.com/file/d/1BcGAgZwAam2zFWWvQ8CDfBGxj3RBU0Hr/view?usp=sharing))
  - TopDial ([GitHub](https://github.com/iwangjian/TopDial))

*注*：对于自定义数据集，你可以参考 `data/dummy_data.json` 来了解数据格式。

## 快速上手

实验中使用的大语言模型（LLMs）均从以下 Hugging Face 模型库下载：

  - [LLaMA-7B](https://huggingface.co/yahma/llama-7b-hf)
  - [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - [Vicuna-7B-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
  - [Llama 2 Chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

假设你已经下载了分词器（tokenizer）和模型权重 `{MODEL_NAME}`，并将其存放在 `pretrained/{MODEL_NAME}` 目录下，你可以运行以下命令进行训练、推理和评估。

### 训练

```bash
deepspeed --master_port=29600 --include="localhost:0,1" src/midituning/finetune.py \
    --model_name_or_path pretrained/${MODEL_NAME} \
    --data_path data/${DATASET_NAME}/data_fmt_dialog/train.json \
    --weight_beta 1.0 \
    --max_instruction_length 256 \
    --max_utterance_length 72 \
    --max_rounds 10 \
    --num_proc 8 \
    --output_dir logs/${DATASET_NAME}/midi_${MODEL_NAME} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --q_lora True \
    --deepspeed config/deepspeed_config_s2.json
```

### 推理

```bash
export CUDA_VISIBLE_DEVICES="0,1"

accelerate launch --main_process_port 29600 \
    --multi_gpu --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    src/midituning/generate.py \
    --model_path logs/${DATASET_NAME}/midi_${MODEL_NAME} \
    --test_data_path data/${DATASET_NAME}/data_fmt_dialog/test.json \
    --test_unseen_data_path data/${DATASET_NAME}/data_fmt_dialog/test_unseen.json \
    --output_dir results/${DATASET_NAME}/midi_${MODEL_NAME}\
    --max_instruction_length 320 \
    --max_utterance_length 100 \
    --max_rounds 10 \
    --max_new_tokens 100 \
    --temperature 0.5 \
    --top_p 0.75 \
    --top_k 40
```

### 评估

对于对话生成常用的自动评估指标，可以运行以下命令：

```bash
# 以 LIGHT 数据集为例
python eval/eval_light.py \
    --eval_file results/light/midi_${MODEL_NAME}/test_output.jsonl \
    --gold_file data/light/light_test.jsonl 

python eval/eval_light.py \
    --eval_file results/light/midi_${MODEL_NAME}/test_unseen_output.jsonl \
    --gold_file data/light/light_test_unseen.jsonl
```

为了测量**一致性概率（consistency probability）**，首先需要从 [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased) 下载 `BERT-base-uncased` 模型，并将所有文件存放在 `pretrained/bert-base-uncased` 目录下。然后，运行以下命令构建一致性评估器：

```bash
# 训练评估器
python src/detector/run.py --data_dir data/${DATASET_NAME} \
    --output_dir logs/${DATASET_NAME}/detector \
    --bert_model pretrained/bert-base-uncased \
    --architecture "detect" \
    --max_length 500 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_steps 500

# 进行评估
python src/detector/run.py --eval --plot --data_dir data/${DATASET_NAME} \
    --output_dir logs/${DATASET_NAME}/detector \
    --bert_model pretrained/bert-base-uncased \
    --max_length 500 \
    --eval_batch_size 32
```

随后，在运行 `eval/eval_light.py` 或 `eval/eval_topdial.py` 时添加 `--detector_model logs/${DATASET_NAME}/detector` 参数，即可计算一致性概率。

若要获取 **GPT-4 评分**，你需要准备一个 OpenAI API 密钥并存入文件中（例如 `openai_api_key.txt`）。然后运行以下命令：

```bash
# 以 LIGHT 数据集为例
python eval/eval_by_gpt.py \
    --eval_file results/light/midi_${MODEL_NAME}/test_output.jsonl \
    --gold_file data/light/light_test.jsonl \
    --prompt_template prompt/eval_light.txt \
    --model "gpt-4-turbo"
```

## 引用

如果你觉得我们的代码对你的研究有所帮助，请引用我们的论文：

```bibtex
@inproceedings{wang-etal-2024-instruct,
  title={Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue},
  author={Wang, Jian and 
      Leong, Chak Tou and 
      Wang, Jiashuo and 
      Lin, Dongding and 
      Li, Wenjie and 
      Wei, Xiao-Yong},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024}
}
```
