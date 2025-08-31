"""
使用 TRL 库对 Qwen2.5-0.5B 进行 LoRA SFT 的训练脚本
"""

import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json


def load_and_prepare_dataset(data_path, tokenizer, max_length=512):
    """
    加载和预处理数据集
    支持 JSON Lines 格式的数据，数据格式为 {"prompt": "...", "chosen": "..."}
    """
    
    # 示例数据格式 - 如果没有提供数据路径，使用示例数据
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在，使用示例数据")
        sample_data = [
            {
                "prompt": "请介绍一下北京的历史",
                "chosen": "北京是中国的首都，有着悠久的历史。作为六朝古都，北京拥有丰富的文化遗产和历史建筑。"
            },
            {
                "prompt": "解释什么是机器学习",
                "chosen": "机器学习是人工智能的一个分支，它让计算机能够在没有明确编程的情况下学习和改进。"
            }
        ]
        dataset = Dataset.from_list(sample_data)
    else:
        # 加载 JSON Lines 格式的数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        dataset = Dataset.from_list(data)
    
    def format_prompt(example):
        """格式化输入提示，将chosen字段重命名为completion以符合SFTTrainer期望"""
        # 重命名字段以符合SFTTrainer的期望
        return {
            "prompt": example["prompt"],
            "completion": example["chosen"]  # 将chosen重命名为completion
        }
    
    # 格式化数据集
    formatted_dataset = dataset.map(format_prompt)
    
    return formatted_dataset


def create_model_and_tokenizer(model_name):
    """
    创建模型和分词器
    """
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（使用 fp16 精度）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # 确保模型和tokenizer关联
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def setup_lora_config():
    """
    设置 LoRA 配置
    """
    lora_config = LoraConfig(
        r=16,                    # LoRA 秩
        lora_alpha=32,           # LoRA 缩放参数
        target_modules=[         # 目标模块
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,        # LoRA dropout
        bias="none",             # 偏置设置
        task_type="CAUSAL_LM",   # 任务类型
    )
    
    return lora_config


def main():
    """
    主训练函数
    """
    
    # 配置参数
    MODEL_NAME = "models/Qwen2.5-0.5B-Instruct"  # 模型名称
    DATA_PATH = "data/rm_dataset_final.jsonl"     # 训练数据路径
    OUTPUT_DIR = "models/qwen2.5-0.5b-lora-sft"  # 输出目录
    MAX_LENGTH = 512                   # 最大序列长度
    
    print("开始加载模型和分词器...")
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
    
    print("设置 LoRA 配置...")
    
    # 设置 LoRA 配置
    lora_config = setup_lora_config()
    
    # 将 LoRA 应用到模型
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    print("加载和处理数据集...")
    
    # 加载数据集
    train_dataset = load_and_prepare_dataset(DATA_PATH, tokenizer, MAX_LENGTH)
    
    print(f"训练数据集大小: {len(train_dataset)}")
    
    # 训练参数 - 使用 SFTConfig 替代 TrainingArguments
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=None,  # 不使用 wandb 等记录工具
        # SFT 特定参数
        max_length=MAX_LENGTH,
        packing=False,  # 不进行序列打包
        dataset_num_proc=2,
    )
    
    print("初始化 SFT 训练器...")
    
    # 创建 SFT 训练器
    trainer = SFTTrainer(
        model=model,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        train_dataset=train_dataset,
        args=sft_config,
    )
    
    print("开始训练...")
    
    # 开始训练
    trainer.train()
    
    print("保存模型...")
    
    # 保存 LoRA 适配器
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"训练完成！模型已保存到: {OUTPUT_DIR}")


def create_sample_data():
    """
    创建示例训练数据文件
    """
    sample_data = [
        {
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条侧重使用反馈的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "油痘肌的夏日救星✨ 这支精华让我告别'反光脸'！\n\n每到夏天，我的T区就像个小型油田🌋 妆后2小时就开始脱妆，毛孔里还能挤出白色角栓...直到遇到原素方程的净颜焕采精华，终于明白什么叫源头控油的科学配方。\n\n🧪核心成分拆解：\n• 10%烟酰胺：不是盲目堆浓度！这个黄金配比既能抑制皮脂腺活跃度，又不会刺激屏障\n• 锌PCA：像吸油纸一样吸附表面油脂，但不会拔干\n• 红没药醇：给躁动的痘痘肌'灭火'，我经期爆痘时厚涂它，红肿消退速度肉眼可见"
        },
        {
            "prompt": "请解释什么是深度学习",
            "chosen": "深度学习是机器学习的一个子领域，它基于人工神经网络，特别是深层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理、语音识别等领域取得了显著成果。"
        },
        {
            "prompt": "如何学习Python编程？",
            "chosen": "学习Python编程建议遵循以下步骤：1. 掌握基础语法和数据类型；2. 练习编写简单程序；3. 学习面向对象编程；4. 探索常用库如NumPy、Pandas等；5. 通过实际项目巩固知识。"
        },
        {
            "prompt": "翻译以下英文：Hello, how are you today?",
            "chosen": "你好，你今天怎么样？"
        }
    ]
    
    with open("data/rm_dataset_final.jsonl", "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("示例数据文件 rm_dataset_final.jsonl 已创建")


if __name__ == "__main__":
    # 创建示例数据（如果需要）
    if not os.path.exists("data/rm_dataset_final.jsonl"):
        print("创建示例训练数据...")
        create_sample_data()
    
    # 运行训练
    main()