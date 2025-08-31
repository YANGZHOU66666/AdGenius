import json
import random
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_generate.rm_gen_rej_prompts_weak import *
from dataset_generate.rm_gen_rej_prompts_exaggerated import *
from dataset_generate.rm_gen_rej_prompts_fake import *

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

def convert_instruction_to_prompt(instruction: str) -> str:
    """将instruction转换为更自然流畅的中文prompt"""
    if "social_media_review" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条侧重使用反馈的小红书文案"
    elif "social_media_educational" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条科普知识的小红书文案"
    elif "social_media_myth_busting" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条破除护肤误区的小红书文案"
    elif "social_media_storytelling" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条个人使用故事的小红书文案"
    elif "ecommerce_long_form" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条详细的电商平台产品介绍文案"
    elif "paid_ad_cta" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条钩子+利益点+行动号召风格的付费广告文案"
    elif "paid_ad_pas" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条痛点-放大-解决风格的付费广告文案"
    elif "paid_ad_bab" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条之前-之后-桥梁风格的付费广告文案"
    elif "paid_ad_fab" in instruction:
        return "请为\"原素方程\"品牌的如下产品创作一条特点-优势-好处风格的付费广告文案"
    else:
        return "请为\"原素方程\"品牌的如下产品创作一条广告文案"

def get_content_type_from_instruction(instruction: str) -> str:
    """从instruction中提取内容类型"""
    if "social_media_review" in instruction:
        return "social_media_review"
    elif "social_media_educational" in instruction:
        return "social_media_educational"
    elif "social_media_myth_busting" in instruction:
        return "social_media_myth_busting"
    elif "social_media_storytelling" in instruction:
        return "social_media_storytelling"
    elif "ecommerce_long_form" in instruction:
        return "ecommerce_long_form"
    elif "paid_ad_cta" in instruction:
        return "paid_ad_cta"
    elif "paid_ad_pas" in instruction:
        return "paid_ad_pas"
    elif "paid_ad_bab" in instruction:
        return "paid_ad_bab"
    elif "paid_ad_fab" in instruction:
        return "paid_ad_fab"
    else:
        return "unknown"

def call_api_with_prompt(prompt: str, model_name: str, client: OpenAI) -> str:
    """调用API生成回答"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1024,
        )
        
        generated_data = json.loads(response.choices[0].message.content)
        output_text = generated_data.get("output", "").strip()
        
        if output_text:
            return output_text
        else:
            return "生成失败，请重试。"
            
    except Exception as e:
        print(f"API调用失败: {e}")
        return f"API调用出错: {e}"

def generate_weak_rejected(input_data: str, content_type: str, client: OpenAI) -> str:
    """生成weak类型的rejected回答"""
    try:
        # 根据内容类型选择对应的prompt函数
        if content_type == "social_media_review":
            prompt = create_weak_review_social_media_prompt(input_data)
        elif content_type == "social_media_educational":
            prompt = create_weak_educational_social_media_prompt(input_data)
        elif content_type == "social_media_myth_busting":
            prompt = create_weak_myth_busting_social_media_prompt(input_data)
        elif content_type == "social_media_storytelling":
            prompt = create_weak_storytelling_social_media_prompt(input_data)
        elif content_type == "ecommerce_long_form":
            prompt = create_weak_ecommerce_prompt(input_data)
        elif content_type == "paid_ad_cta":
            prompt = create_weak_paid_ad_prompt_cta(input_data)
        elif content_type == "paid_ad_pas":
            prompt = create_weak_paid_ad_prompt_pas(input_data)
        elif content_type == "paid_ad_bab":
            prompt = create_weak_paid_ad_prompt_bab(input_data)
        elif content_type == "paid_ad_fab":
            prompt = create_weak_paid_ad_prompt_fab(input_data)
        else:
            # 默认使用ecommerce类型
            prompt = create_weak_ecommerce_prompt(input_data)
        
        # 调用Qwen2.5-7B-Instruct模型生成回答
        return call_api_with_prompt(prompt, "qwen/Qwen2.5-7B-Instruct", client)
        
    except Exception as e:
        print(f"生成weak rejected时出错: {e}")
        return "这款产品很不错，推荐给大家。"

def generate_exaggerated_rejected(input_data: str, content_type: str, client: OpenAI) -> str:
    """生成exaggerated类型的rejected回答"""
    try:
        # 根据内容类型选择对应的prompt函数
        if content_type == "social_media_review":
            prompt = create_exaggerated_review_social_media_prompt(input_data)
        elif content_type == "social_media_educational":
            prompt = create_exaggerated_educational_social_media_prompt(input_data)
        elif content_type == "social_media_myth_busting":
            prompt = create_exaggerated_myth_busting_social_media_prompt(input_data)
        elif content_type == "social_media_storytelling":
            prompt = create_exaggerated_storytelling_social_media_prompt(input_data)
        elif content_type == "ecommerce_long_form":
            prompt = create_exaggerated_ecommerce_prompt(input_data)
        elif content_type == "paid_ad_cta":
            prompt = create_exaggerated_paid_ad_prompt_cta(input_data)
        elif content_type == "paid_ad_pas":
            prompt = create_exaggerated_paid_ad_prompt_pas(input_data)
        elif content_type == "paid_ad_bab":
            prompt = create_exaggerated_paid_ad_prompt_bab(input_data)
        elif content_type == "paid_ad_fab":
            prompt = create_exaggerated_paid_ad_prompt_fab(input_data)
        else:
            # 默认使用ecommerce类型
            prompt = create_exaggerated_ecommerce_prompt(input_data)
        
        # 调用DeepSeek-V3模型生成回答
        return call_api_with_prompt(prompt, "deepseek-ai/DeepSeek-V3", client)
        
    except Exception as e:
        print(f"生成exaggerated rejected时出错: {e}")
        return "这款产品太神奇了！用了就能年轻十岁！"

def generate_fake_rejected(input_data: str, correct_output: str, client: OpenAI) -> str:
    """生成fake类型的rejected回答"""
    try:
        prompt = create_factual_error_prompt(input_data, correct_output)
        # 调用DeepSeek-V3模型生成回答
        return call_api_with_prompt(prompt, "deepseek-ai/DeepSeek-V3", client)
    except Exception as e:
        print(f"生成fake rejected时出错: {e}")
        return "这款产品含15%烟酰胺，适合所有肤质使用。"

def generate_rejected_answer(input_data: str, correct_output: str, content_type: str, client: OpenAI) -> Tuple[str, str]:
    """根据概率分布生成rejected回答，返回(回答, 方法)的元组"""
    rand = random.random()
    
    if rand < 0.5:  # 50%概率选择weak
        answer = generate_weak_rejected(input_data, content_type, client)
        return answer, "weak"
    elif rand < 0.8:  # 30%概率选择exaggerated
        answer = generate_exaggerated_rejected(input_data, content_type, client)
        return answer, "exaggerated"
    else:  # 20%概率选择fake
        answer = generate_fake_rejected(input_data, correct_output, client)
        return answer, "fake"

def process_sft_data_to_rm_data(sft_data: List[Dict], client: OpenAI) -> List[Dict]:
    """将SFT数据转换为RM数据"""
    rm_data = []
    
    for i, item in enumerate(sft_data):
        print(f"正在处理第 {i+1}/{len(sft_data)} 条数据...")
        
        instruction = item["instruction"]
        input_data = item["input"]
        correct_output = item["output"]
        
        # 转换instruction为更自然的prompt
        natural_prompt = convert_instruction_to_prompt(instruction)
        
        # 拼接input到prompt
        full_prompt = f"{natural_prompt}\n\n{input_data}"
        
        # 获取内容类型
        content_type = get_content_type_from_instruction(instruction)
        
        # 生成rejected回答和方法
        rejected_answer, method = generate_rejected_answer(input_data, correct_output, content_type, client)
        
        # 创建RM数据条目
        rm_item = {
            "prompt": full_prompt,
            "chosen": correct_output,
            "rejected": rejected_answer,
            "method": method
        }
        
        rm_data.append(rm_item)
        
        # 添加延迟避免API限制
        time.sleep(1)
    
    return rm_data

def main():
    """主函数"""
    print("开始生成奖励模型训练数据...")
    
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
    
    # 加载SFT数据
    sft_file_path = "data/sft_dataset_cleaned.jsonl"
    print(f"正在加载SFT数据: {sft_file_path}")
    
    try:
        sft_data = load_sft_data(sft_file_path)
        print(f"成功加载 {len(sft_data)} 条SFT数据")
    except Exception as e:
        print(f"加载SFT数据失败: {e}")
        return
    
    # 转换为RM数据
    print("正在生成rejected回答...")
    rm_data = process_sft_data_to_rm_data(sft_data, client)
    print(f"成功生成 {len(rm_data)} 条RM数据")
    
    # 保存为CSV文件
    output_file = "data/rm_dataset.csv"
    print(f"正在保存到: {output_file}")
    
    try:
        df = pd.DataFrame(rm_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"成功保存RM数据集到: {output_file}")
        
        # 显示前几行数据作为预览
        print("\n数据预览:")
        print(df.head())
        
        # 显示数据统计
        print(f"\n数据集统计:")
        print(f"总行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        
    except Exception as e:
        print(f"保存数据失败: {e}")

if __name__ == "__main__":
    main()
