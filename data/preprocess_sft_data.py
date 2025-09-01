import json
from datasets import load_dataset
import os

# --- 配置输入和输出文件 ---
source_file = "data/rm_dataset_final.jsonl"  # 您原始的数据文件，包含prompt, chosen, rejected
output_file = "data/sft_dataset_final.jsonl" # 预处理后用于SFT训练的新文件

def transform_to_messages_format(example):
    """
    将单个数据点从 {prompt, chosen} 格式转换为 SFTTrainer 喜欢的 messages 格式。
    """
    # 我们只需要 prompt 和 chosen (好的回答) 来进行监督微调
    prompt_text = example.get("prompt")
    chosen_text = example.get("chosen")

    if not prompt_text or not chosen_text:
        return None # 如果缺少关键字段，则跳过该行

    # 构建 messages 列表
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": chosen_text}
    ]
    
    return {"messages": messages}

def main():
    """
    主函数，加载原始数据，进行转换，并保存为新文件。
    """
    print(f"--- 开始预处理数据 ---")
    
    if not os.path.exists(source_file):
        print(f"错误: 源文件 '{source_file}' 不存在。请检查文件名和路径。")
        # 创建一个示例文件以便脚本可以运行
        print("为了演示，正在创建一个示例 train.jsonl 文件...")
        sample_data = {
            "prompt": "请为\"原素方程\"品牌的\"净颜焕采精华\"创作一条小红书文案",
            "chosen": "油痘肌的夏日救星✨ 这支精华让我告别‘反光脸’！...",
            "rejected": "..."
        }
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')

    print(f"正在从 '{source_file}' 加载数据...")
    dataset = load_dataset("json", data_files=source_file, split="train")
    
    print("正在将数据转换为 messages 格式...")
    # 使用 .map() 方法高效地应用转换函数
    processed_dataset = dataset.map(transform_to_messages_format, remove_columns=list(dataset.features))
    
    print(f"正在将处理后的数据保存到 '{output_file}'...")
    # 将处理后的数据集保存为 jsonl 文件
    processed_dataset.to_json(output_file, orient="records", force_ascii=False)
    
    print(f"--- 数据预处理完成 ---")
    print(f"✅ 成功生成SFT训练文件: {output_file}")
    print("现在您可以运行模型训练脚本了。")

if __name__ == "__main__":
    main()