import json
import sys
import os
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_val_data(file_path: str) -> List[Dict]:
    """加载验证数据集"""
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

def process_val_data_to_final_data(val_data: List[Dict]) -> List[Dict]:
    """将验证数据转换为最终格式"""
    final_data = []
    
    for i, item in enumerate(val_data):
        print(f"正在处理第 {i+1}/{len(val_data)} 条数据...")
        
        instruction = item["instruction"]
        input_data = item["input"]
        output = item["output"]
        
        # 转换instruction为更自然的prompt
        natural_prompt = convert_instruction_to_prompt(instruction)
        
        # 获取内容类型
        content_type = get_content_type_from_instruction(instruction)
        
        # 拼接input到prompt
        full_prompt = f"{natural_prompt}\n\n{input_data}"
        
        # 创建最终数据条目
        final_item = {
            "prompt": full_prompt,
            "chosen": output,
            "type": content_type
        }
        
        final_data.append(final_item)
    
    return final_data

def save_final_data(final_data: List[Dict], output_file: str):
    """保存最终数据到JSONL文件"""
    print(f"正在保存到: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"成功保存验证数据集到: {output_file}")
        print(f"总数据条数: {len(final_data)}")
        
        # 显示前几行数据作为预览
        print("\n数据预览:")
        for i, item in enumerate(final_data[:3]):
            print(f"第{i+1}条:")
            print(f"  Prompt: {item['prompt'][:100]}...")
            print(f"  Chosen: {item['chosen'][:100]}...")
            print()
        
    except Exception as e:
        print(f"保存数据失败: {e}")

def main():
    """主函数"""
    print("开始预处理验证数据...")
    
    # 加载验证数据
    val_file_path = "data/val_dataset_cleaned_cta.jsonl"
    print(f"正在加载验证数据: {val_file_path}")
    
    try:
        val_data = load_val_data(val_file_path)
        print(f"成功加载 {len(val_data)} 条验证数据")
    except Exception as e:
        print(f"加载验证数据失败: {e}")
        return
    
    # 转换为最终格式
    print("正在处理数据格式...")
    final_data = process_val_data_to_final_data(val_data)
    print(f"成功处理 {len(final_data)} 条数据")
    
    # 保存为最终文件
    output_file = "data/val_dataset_final.jsonl"
    save_final_data(final_data, output_file)

if __name__ == "__main__":
    main()
