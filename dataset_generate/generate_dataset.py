import pandas as pd
import json
import time
from tqdm import tqdm
import os
from openai import OpenAI

# 从prompts.py文件中导入我们设计的Prompt函数
from prompts import (
    # 社交媒体 (4种)
    create_social_media_prompt_review,
    create_social_media_prompt_educational,
    create_social_media_prompt_myth_busting,
    create_social_media_prompt_storytelling,
    # 电商 (1种)
    create_ecommerce_prompt,
    # 付费广告 (4种)
    create_paid_ad_prompt_cta,
    create_paid_ad_prompt_pas,
    create_paid_ad_prompt_bab,
    create_paid_ad_prompt_fab
)

# --- 1. 配置 ---
# 在这里填入您的硅基流动API密钥
# 强烈建议使用环境变量以保证安全: api_key = os.getenv("SILICONFLOW_API_KEY")
# 如果直接填入，请注意不要泄露您的密钥:
SILICONFLOW_API_KEY = "sk-qadtenuyxrfcbbsrwlmevvozivtwptlefipuqnoirdynnovz"

# --- 使用说明 ---
# 1. 首先安装依赖：pip install pandas tqdm openai
# 2. 确保 data/products.csv 文件存在
# 3. 运行 products_process.py 生成 products_processed.csv
# 4. 运行此脚本生成数据集

# 输入和输出文件名
PRODUCT_CSV_PATH = "data/products_processed.csv"
OUTPUT_JSONL_PATH = "data/sft_dataset_deepseek.jsonl"

# 硅基流动平台上的模型名称
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

# --- 2. 主执行逻辑 ---

def check_dependencies():
    """检查必要的依赖包是否已安装"""
    missing_packages = []

    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")

    try:
        import tqdm
    except ImportError:
        missing_packages.append("tqdm")

    try:
        import openai
    except ImportError:
        missing_packages.append("openai")

    if missing_packages:
        print("错误：缺少以下依赖包：")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def main():
    """主函数，执行数据集生成流程"""
    print("--- SFT数据集生成脚本启动 (使用硅基流动API) ---")

    # 检查依赖
    if not check_dependencies():
        return

    # 检查API密钥
    if SILICONFLOW_API_KEY == "sk-YOUR_API_KEY_HERE" or not SILICONFLOW_API_KEY:
        print("错误：请在脚本中填入您的硅基流动API密钥。")
        return

    # 初始化硅基流动客户端
    try:
        client = OpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key=SILICONFLOW_API_KEY
        )
    except Exception as e:
        print(f"初始化API客户端时出错: {e}")
        return

    # 检查产品文件是否存在
    if not os.path.exists(PRODUCT_CSV_PATH):
        print(f"错误：找不到产品文件 '{PRODUCT_CSV_PATH}'。")
        print(f"当前工作目录：{os.getcwd()}")
        print(f"请确保产品文件存在，或者先运行 products_process.py 处理数据")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_JSONL_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录：{output_dir}")

    # 读取产品数据
    try:
        df = pd.read_csv(PRODUCT_CSV_PATH)
        print(f"成功读取 {len(df)} 条产品信息。")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    sft_dataset = []

    prompt_functions = {
        "social_media_review": create_social_media_prompt_review,
        "social_media_educational": create_social_media_prompt_educational,
        "social_media_myth_busting": create_social_media_prompt_myth_busting,
        "social_media_storytelling": create_social_media_prompt_storytelling,
        "ecommerce_long_form": create_ecommerce_prompt,
        "paid_ad_cta": create_paid_ad_prompt_cta,
        "paid_ad_pas": create_paid_ad_prompt_pas,
        "paid_ad_bab": create_paid_ad_prompt_bab,
        "paid_ad_fab": create_paid_ad_prompt_fab
    }

    total_tasks = len(df) * len(prompt_functions)
    with tqdm(total=total_tasks, desc="生成SFT数据") as pbar:
        for index, row in df.iterrows():
            product_info_dict = row.to_dict()
            product_info_str = json.dumps(product_info_dict, ensure_ascii=False, indent=2)

            for task_type, create_prompt_func in prompt_functions.items():
                prompt = create_prompt_func(product_info_str)

                try:
                    # 调用硅基流动API
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}, # 确保输出为JSON
                        temperature=0.7,
                        max_tokens=1024,
                    )

                    generated_data = json.loads(response.choices[0].message.content)
                    output_text = generated_data.get("output", "").strip()

                    if output_text:
                        sft_record = {
                            "instruction": f"为‘原素方程’品牌生成一条“{task_type}”类型的广告文案。",
                            "input": json.dumps(product_info_dict, ensure_ascii=False),
                            "output": output_text
                        }
                        sft_dataset.append(sft_record)

                    time.sleep(1)

                except Exception as e:
                    print(f"\n处理产品 '{product_info_dict.get('product_name')}' (任务: {task_type}) 时出错: {e}")

                pbar.update(1)

    # 保存最终数据集
    try:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
            for record in sft_dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n--- 任务完成 ---")
        print(f"成功生成 {len(sft_dataset)} 条SFT数据，已保存至 '{OUTPUT_JSONL_PATH}'。")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()