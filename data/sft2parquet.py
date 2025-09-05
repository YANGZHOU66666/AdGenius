import json
import pandas as pd
from typing import Dict, List, Any

def convert_sft_to_parquet(input_file: str, output_file: str):
    """
    将SFT数据集JSONL文件转换为Parquet格式
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的Parquet文件路径
    """
    data_list = []
    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data = json.loads(line.strip())
                
                # 提取用户输入prompt（messages中的user角色内容）
                user_prompt = None
                assistant_response = None
                
                for message in data['messages']:
                    if message['role'] == 'user':
                        user_prompt = message['content']
                    elif message['role'] == 'assistant':
                        assistant_response = message['content']
                
                # 构建Hugging Face template格式的prompt
                # 只包含用户输入，格式为[{content:..., role: user}]
                prompt = [{"content": user_prompt, "role": "user"}]
                
                # 构建reward_model字典
                reward_model = {
                    "style": "model",
                    "ground_truth": assistant_response
                }
                
                # 构建最终数据行
                row = {
                    "data_source": "elemental_equation",
                    "prompt": prompt,
                    "ability": "style_imitation", 
                    "reward_model": reward_model,
                    "extra_info": None  # 使用None而不是空字典，避免PyArrow错误
                }
                
                data_list.append(row)
    
    # 创建DataFrame并保存为Parquet
    df = pd.DataFrame(data_list)
    df.to_parquet(output_file, index=False)
    print(f"成功转换 {len(data_list)} 条数据到 {output_file}")

if __name__ == "__main__":
    # 转换SFT数据集
    input_file = "data/sft_dataset_final.jsonl"
    output_file = "data/sft_dataset_final.parquet"
    
    convert_sft_to_parquet(input_file, output_file)
