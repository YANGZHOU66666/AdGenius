import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    # === 1. 配置路径 ===
    # 您的原始基础模型路径 (与训练时使用的模型一致)
    base_model_path = "models/Qwen2.5-0.5B-Instruct" 
    # 您训练好的LoRA适配器路径
    adapter_path = "models/qwen2_5-0.5b-sft-lora-adapter-test" 
    # 您希望保存合并后模型的路径
    merged_model_path = "models/qwen2_5-0.5b-instruct-skincare-merged"

    print("--- 开始合并 LoRA 适配器 ---")

    # === 2. 加载基础模型和Tokenizer ===
    print(f"正在从 '{base_model_path}' 加载基础模型...")
    # 以bfloat16精度加载，确保与训练时一致
    # 注意：这里不能使用4-bit量化 (不能有 quantization_config)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", # 自动选择设备 (GPU或CPU)
    )

    print(f"正在从 '{base_model_path}' 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # === 3. 加载并应用LoRA适配器 ===
    print(f"正在从 '{adapter_path}' 加载并应用LoRA适配器...")
    # PeftModel会自动将适配器权重加载到基础模型之上
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # === 4. 执行合并！ ===
    print(">>> 正在执行合并操作...")
    # merge_and_unload() 会将LoRA权重合并到基础模型中，并返回合并后的新模型
    merged_model = peft_model.merge_and_unload()
    print(">>> 合并完成！")

    # === 5. 保存合并后的完整模型和Tokenizer ===
    print(f"正在将合并后的模型保存到 '{merged_model_path}'...")
    os.makedirs(merged_model_path, exist_ok=True)
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    print(f"✅ 合并后的模型已成功保存至: {merged_model_path}")

if __name__ == "__main__":
    main()