import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding
import os
from typing import List, Dict, Any


def load_rm_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 rm_dataset_final.jsonl 文件并转换为列表格式
    
    Args:
        file_path: jsonl 文件路径
        
    Returns:
        包含训练数据的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line.strip()))
    return data


def convert_to_hf_format(data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    将原始数据转换为 Hugging Face 格式
    
    Args:
        data: 原始数据列表
        
    Returns:
        符合 Hugging Face 格式的字典
    """
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for item in data:
        prompts.append(item['prompt'])
        chosen_responses.append(item['chosen'])
        rejected_responses.append(item['rejected'])
    
    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses
    }


def finetune_reward_model():
    """
    完整的奖励模型微调脚本
    """
    print("=== 开始奖励模型微调 ===")
    
    # --- 配置路径 ---
    # 可以根据具体情况修改以下路径
    model_path = "models/Skywork-Reward-V2-Qwen3-0.6B"  # 基础模型路径
    output_dir = "models/Skywork-Reward-V2-Qwen3-0.6B-finetuned"  # 输出模型路径
    dataset_path = "data/rm_dataset_final.jsonl"  # 数据集路径
    
    # --- 第一步：准备数据集 ---
    print("\n--- 1. 准备数据集中... ---")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    # 读取原始数据
    raw_data = load_rm_dataset(dataset_path)
    print(f"成功读取 {len(raw_data)} 条训练数据")
    
    # 转换为 Hugging Face 格式
    train_data = convert_to_hf_format(raw_data)
    train_dataset = Dataset.from_dict(train_data)
    print("--- 数据集准备完成 ---")

    # --- 第二步：加载基础模型和分词器 ---
    print(f"\n--- 2. 加载基础模型: {model_path} ---")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型，这是一个序列分类模型，并且我们告诉它只有一个输出分数 (num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=1, 
        trust_remote_code=True
    )

    # 检查是否有可用的 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"--- 模型加载成功，将运行在: {device} ---")

    # --- 第三步：数据预处理 ---
    print("\n--- 3. 数据预处理中... ---")
    
    def formatting_func(examples):
        """
        将文本数据转换成模型可以理解的 token ID
        """
        # 定义分词器的参数
        kwargs = {
            "padding": "max_length", 
            "truncation": True, 
            "max_length": 1024,  # 增加长度以容纳更长的文案
            "return_tensors": None
        }
        
        # 1. 首先，将 chosen 对话应用聊天模板，生成格式化的文本列表
        chosen_texts = []
        for prompt, response in zip(examples["prompt"], examples["chosen"]):
            messages = [
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content": response}
            ]
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                chosen_texts.append(formatted_text)
            except Exception as e:
                # 如果聊天模板失败，使用简单的格式
                chosen_texts.append(f"用户: {prompt}\n助手: {response}")
        
        # 2. 然后，将 rejected 对话也生成格式化的文本列表
        rejected_texts = []
        for prompt, response in zip(examples["prompt"], examples["rejected"]):
            messages = [
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content": response}
            ]
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                rejected_texts.append(formatted_text)
            except Exception as e:
                # 如果聊天模板失败，使用简单的格式
                rejected_texts.append(f"用户: {prompt}\n助手: {response}")

        # 3. 最后，对格式化好的文本列表进行批量分词
        tokens_chosen = tokenizer(chosen_texts, **kwargs)
        tokens_rejected = tokenizer(rejected_texts, **kwargs)

        # 返回一个包含 token ID 和 attention mask 列表的字典
        return {
            "input_ids_chosen": tokens_chosen["input_ids"],
            "attention_mask_chosen": tokens_chosen["attention_mask"],
            "input_ids_rejected": tokens_rejected["input_ids"],
            "attention_mask_rejected": tokens_rejected["attention_mask"],
        }
    
    # 对数据集应用这个格式化函数
    formatted_dataset = train_dataset.map(formatting_func, batched=True)
    print("--- 数据预处理完成 ---")

    # --- 第四步：配置并开始训练 ---
    print("\n--- 4. 配置训练参数并开始训练... ---")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,           # 根据显存调整批次大小
        num_train_epochs=1,                      # 训练轮次
        learning_rate=5e-6,                      # 较小的学习率
        logging_steps=10,                        # 每隔多少步打印一次日志
        save_steps=20,                          # 每隔多少步保存一次模型
        eval_steps=20,                          # 每隔多少步评估一次
        warmup_steps=50,                         # 预热步数
        remove_unused_columns=False,             # 必须设置为 False
        dataloader_drop_last=True,               # 丢弃最后一个不完整的批次
        save_total_limit=3,                      # 最多保存3个检查点
        load_best_model_at_end=True,             # 训练结束时加载最佳模型
        metric_for_best_model="loss",            # 用于选择最佳模型的指标
        greater_is_better=False,                 # loss越小越好
        report_to=None,                          # 不使用wandb等工具
    )

    # 初始化 RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
    )

    # 开始训练！
    print("开始训练...")
    trainer.train()
    print("--- 训练完成！---")

    # --- 第五步：保存模型 ---
    print("\n--- 5. 保存微调后的模型... ---")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"--- 模型已成功保存到 '{output_dir}' 目录 ---")
    
    # 保存训练配置信息
    config_info = {
        "base_model": model_path,
        "dataset_size": len(raw_data),
        "training_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "max_length": 1024
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)
    
    print("--- 训练配置信息已保存 ---")
    print("=== 奖励模型微调完成 ===")


if __name__ == "__main__":
    try:
        finetune_reward_model()
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
