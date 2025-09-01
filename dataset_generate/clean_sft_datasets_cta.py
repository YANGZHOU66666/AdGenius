import json
import os
import sys
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_generate.sft_clean_prompts_cta import get_paid_ad_cleaning_prompt

# 配置API密钥
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

def load_sft_data(file_path: str) -> List[Dict]:
    """加载SFT数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def is_paid_ad_type(instruction: str) -> bool:
    """判断是否为paid_ad类型"""
    paid_ad_types = ["paid_ad_cta", "paid_ad_pas", "paid_ad_bab", "paid_ad_fab"]
    return any(ad_type in instruction for ad_type in paid_ad_types)

def clean_paid_ad_with_api(ad_copy: str, client: OpenAI) -> str:
    """使用API清洗付费广告文案，确保包含CTA"""
    try:
        # 获取清洗prompt
        cleaning_prompt = get_paid_ad_cleaning_prompt(ad_copy)
        
        # 调用API进行清洗
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": cleaning_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # 使用较低的温度确保一致性
            max_tokens=1024,
        )
        
        # 解析API响应
        result = json.loads(response.choices[0].message.content)
        cleaned_output = result.get("cleaned_output", ad_copy)
        
        return cleaned_output
        
    except Exception as e:
        print(f"API清洗失败: {e}")
        # 如果API调用失败，返回原文
        return ad_copy

def clean_sft_dataset(sft_data: List[Dict], client: OpenAI) -> List[Dict]:
    """清洗SFT数据集"""
    cleaned_data = []
    
    for i, item in enumerate(sft_data):
        instruction = item["instruction"]
        input_data = item["input"]
        output = item["output"]
        
        print(f"正在处理第 {i+1}/{len(sft_data)} 条数据...")
        
        # 如果是paid_ad类型，进行CTA清洗
        if is_paid_ad_type(instruction):
            print(f"  检测到paid_ad类型，进行CTA清洗...")
            cleaned_output = clean_paid_ad_with_api(output, client)
            
            # 创建清洗后的数据条目
            cleaned_item = {
                "instruction": instruction,
                "input": input_data,
                "output": cleaned_output
            }
            
            # 如果输出有变化，显示对比
            if cleaned_output != output:
                print(f"  原文: {output[:100]}...")
                print(f"  清洗后: {cleaned_output[:100]}...")
            else:
                print(f"  无需修改，已包含清晰CTA")
                
        else:
            # 非paid_ad类型，直接复制
            print(f"  非paid_ad类型，直接复制")
            cleaned_item = item.copy()
        
        cleaned_data.append(cleaned_item)
        
        # 添加延迟避免API限制
        import time
        time.sleep(1)
    
    return cleaned_data

def save_cleaned_data(cleaned_data: List[Dict], output_path: str):
    """保存清洗后的数据"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存为JSONL格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"成功保存清洗后的数据到: {output_path}")
        
    except Exception as e:
        print(f"保存数据失败: {e}")

def main():
    """主函数"""
    print("开始清洗SFT数据集，确保paid_ad类型包含CTA...")
    
    # 检查API密钥
    if not SILICONFLOW_API_KEY or SILICONFLOW_API_KEY == "sk-YOUR_API_KEY_HERE":
        print("错误：请在.env文件中设置SILICONFLOW_API_KEY")
        return
    
    # 初始化API客户端
    try:
        client = OpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key=SILICONFLOW_API_KEY
        )
        print("成功初始化API客户端")
    except Exception as e:
        print(f"初始化API客户端失败: {e}")
        return
    
    # 输入和输出文件路径
    input_file = "data/val_dataset_cleaned.jsonl"
    output_file = "data/val_dataset_cleaned_cta.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 '{input_file}'")
        return
    
    # 加载原始数据
    print(f"正在加载原始数据: {input_file}")
    try:
        sft_data = load_sft_data(input_file)
        print(f"成功加载 {len(sft_data)} 条数据")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 统计paid_ad类型数量
    paid_ad_count = sum(1 for item in sft_data if is_paid_ad_type(item["instruction"]))
    print(f"检测到 {paid_ad_count} 条paid_ad类型数据需要清洗")
    
    # 清洗数据
    print("开始清洗数据...")
    cleaned_data = clean_sft_dataset(sft_data, client)
    
    # 保存清洗后的数据
    print("保存清洗后的数据...")
    save_cleaned_data(cleaned_data, output_file)
    
    # 显示统计信息
    print(f"\n=== 清洗完成 ===")
    print(f"原始数据条数: {len(sft_data)}")
    print(f"清洗后数据条数: {len(cleaned_data)}")
    print(f"paid_ad类型数据条数: {paid_ad_count}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    main()
