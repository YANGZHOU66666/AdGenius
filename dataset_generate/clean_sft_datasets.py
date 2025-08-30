import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入清洗提示词
from sft_clean_prompts import create_clean_prompt

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
API_BASE_URL = "https://api.siliconflow.cn/v1"
INPUT_JSONL_PATH = "data/val_dataset_deepseek.jsonl"
OUTPUT_JSONL_PATH = "data/val_dataset_cleaned.jsonl"

# --- 主执行逻辑 ---

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

    try:
        import dotenv
    except ImportError:
        missing_packages.append("python-dotenv")

    if missing_packages:
        print("错误：缺少以下依赖包：")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def validate_config():
    """验证配置"""
    if not SILICONFLOW_API_KEY:
        raise ValueError(
            "缺少必要的环境变量 SILICONFLOW_API_KEY。\n"
            "请创建 .env 文件并添加：SILICONFLOW_API_KEY=your_api_key_here"
        )
    return True

def clean_single_record(client, record, task_type):
    """清洗单条SFT数据记录"""
    try:
        # 创建清洗提示词
        prompt = create_clean_prompt(json.dumps(record, ensure_ascii=False))
        
        # 调用API进行清洗
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # 使用较低的温度确保一致性
            max_tokens=1024,
        )
        
        # 解析响应
        generated_data = json.loads(response.choices[0].message.content)
        cleaned_output = generated_data.get("output", "").strip()
        
        if cleaned_output:
            # 创建清洗后的记录，保留instruction和input字段
            cleaned_record = {
                "instruction": record["instruction"],
                "input": record["input"],
                "output": cleaned_output
            }
            return cleaned_record, True
        else:
            print(f"警告：任务 {task_type} 返回空输出")
            return record, False
            
    except Exception as e:
        print(f"清洗任务 {task_type} 时出错: {e}")
        return record, False

def main():
    """主函数，执行数据集清洗流程"""
    print("--- SFT数据集清洗脚本启动 ---")

    # 检查依赖
    if not check_dependencies():
        return

    # 验证配置
    try:
        validate_config()
    except ValueError as e:
        print(f"配置错误: {e}")
        return

    # 初始化API客户端
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=SILICONFLOW_API_KEY
        )
    except Exception as e:
        print(f"初始化API客户端时出错: {e}")
        return

    # 检查输入文件是否存在
    if not os.path.exists(INPUT_JSONL_PATH):
        print(f"错误：找不到输入文件 '{INPUT_JSONL_PATH}'。")
        print(f"当前工作目录：{os.getcwd()}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_JSONL_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录：{output_dir}")

    # 读取原始数据集
    try:
        with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
            original_records = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"成功读取 {len(original_records)} 条原始数据。")
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return

    # 清洗数据集
    cleaned_records = []
    success_count = 0
    error_count = 0

    print("开始清洗数据...")
    with tqdm(total=len(original_records), desc="清洗SFT数据") as pbar:
        for i, record in enumerate(original_records):
            # 提取任务类型用于日志
            instruction = record.get("instruction", "")
            task_type = "unknown"
            if "social_media" in instruction:
                task_type = "social_media"
            elif "ecommerce" in instruction:
                task_type = "ecommerce"
            elif "paid_ad" in instruction:
                task_type = "paid_ad"
            
            # 清洗单条记录
            cleaned_record, success = clean_single_record(client, record, f"{task_type}_{i}")
            
            if success:
                cleaned_records.append(cleaned_record)
                success_count += 1
            else:
                # 如果清洗失败，保留原始记录
                cleaned_records.append(record)
                error_count += 1
            
            # 更新进度条
            pbar.update(1)
            
            # 添加延迟避免API限制
            time.sleep(0.5)

    # 保存清洗后的数据集
    try:
        with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\n--- 任务完成 ---")
        print(f"成功清洗 {success_count} 条数据")
        print(f"清洗失败 {error_count} 条数据（保留原始内容）")
        print(f"清洗后数据集已保存至 '{OUTPUT_JSONL_PATH}'")
        print(f"总计处理 {len(cleaned_records)} 条数据")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main()