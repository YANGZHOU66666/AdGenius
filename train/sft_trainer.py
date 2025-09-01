import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

def main():
    # === 1. 配置模型、数据和输出路径 ===
    # 模型ID
    base_model_name = "models/Qwen2.5-0.5B-Instruct"
    # 预处理好的数据集路径
    dataset_path = "data/sft_dataset_final.jsonl"
    # LoRA适配器输出目录
    output_dir = "models/qwen2_5-0.5b-sft-lora-adapter-test-2"

    # === 2. 加载数据集 ===
    print(f"正在从 '{dataset_path}' 加载数据集...")
    # 确保文件存在，否则创建一个示例
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件 '{dataset_path}' 不存在。请先运行 preprocess_sft_data.py。")
        return
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"数据集长度: {len(dataset)}")

    # === 3. 配置LoRA (PEFT) ===
    # 这是官方文档推荐的与SFTTrainer结合使用的方法
    print(">>> 正在配置LoRA...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # === 4. 配置训练参数 (严格使用SFTConfig) ===
    print(">>> 正在配置SFTConfig训练参数...")
    training_args = SFTConfig(
        # --- 核心训练参数 ---
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # 由于显存占用大，从1开始
        gradient_accumulation_steps=8,  # 等效batch_size = 1 * 8 = 8
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        # optim="paged_adamw_8bit", # 依然推荐，可以节省一些显存

        # --- 模型加载参数 (通过model_init_kwargs传递) ---
        # 这是官方文档推荐的加载方式，而不是在外部加载模型
        model_init_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},

        # --- SFTTrainer特有参数 ---
        max_length=2048,
        packing=False, # 启用packing提升效率

        # --- 日志和保存 ---
        logging_steps=20,
        save_strategy="epoch",
        
        # --- 精度相关 ---
        bf16=True, # 在支持的硬件上启用bf16
    )

    # === 5. 初始化并开始训练 ===
    print(">>> 正在初始化SFT Trainer...")
    # 注意：现在我们将模型ID（字符串）直接传给Trainer
    # Trainer会使用SFTConfig中的model_init_kwargs来加载模型
    trainer = SFTTrainer(
        model=base_model_name,
        args=training_args,
        train_dataset=dataset,
        # tokenizer在未提供时，会自动从模型ID加载
        peft_config=peft_config,
        # dataset_text_field="messages", # 当列名为'messages'时，trainer会自动识别，通常无需指定
    )
    
    # 检查并设置pad_token (一个好的实践)
    if trainer.tokenizer.pad_token is None:
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

    print(">>> 开始训练...")
    trainer.train()

    print(">>> 训练完成，正在保存最终的LoRA适配器...")
    trainer.save_model(output_dir)

    print(f"✅ LoRA适配器已成功保存至: {output_dir}")

if __name__ == "__main__":
    main()