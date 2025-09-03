import pandas as pd
import json
import time
from tqdm import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# 从judge_prompts_new.py文件中导入评测Prompt函数
from judge_prompts_new import (
    get_social_media_judge_prompt,
    get_paid_ad_judge_prompt,
    get_ecommerce_judge_prompt
)

# --- 配置 ---
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 输入和输出文件名
EVALUATION_CSV_PATH = "results/evaluation_results_final.csv"  # 评测结果CSV文件路径
VAL_DATASET_JSONL_PATH = "data/val_dataset_final.jsonl"  # 验证数据集路径
OUTPUT_CSV_PATH = "results/single_model_judge_results.csv"  # 输出CSV文件路径
RESPONSE_KEY = "finetuned_model_output"  # 模型回答的键名

# 硅基流动平台上的模型名称
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

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

def load_data():
    """加载评测数据和验证数据集"""
    print("正在加载数据...")
    
    # 加载评测结果CSV
    if not os.path.exists(EVALUATION_CSV_PATH):
        print(f"错误：找不到评测结果文件 '{EVALUATION_CSV_PATH}'")
        return None, None
    
    evaluation_df = pd.read_csv(EVALUATION_CSV_PATH)
    print(f"成功加载评测数据: {len(evaluation_df)} 条")
    
    # 加载验证数据集JSONL
    if not os.path.exists(VAL_DATASET_JSONL_PATH):
        print(f"错误：找不到验证数据集文件 '{VAL_DATASET_JSONL_PATH}'")
        return None, None
    
    val_data = []
    with open(VAL_DATASET_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))
    
    print(f"成功加载验证数据: {len(val_data)} 条")
    
    return evaluation_df, val_data

def get_judge_prompt_by_type(content_type, prompt, model_output):
    """根据内容类型选择对应的评测prompt"""
    if content_type in ["social_media_review", "social_media_educational", "social_media_myth_busting", "social_media_storytelling"]:
        return get_social_media_judge_prompt(prompt, model_output)
    elif content_type in ["paid_ad_cta", "paid_ad_pas", "paid_ad_bab", "paid_ad_fab"]:
        return get_paid_ad_judge_prompt(prompt, model_output)
    elif content_type == "ecommerce_long_form":
        return get_ecommerce_judge_prompt(prompt, model_output)
    else:
        print(f"警告：未知的内容类型 '{content_type}'，使用社交媒体评测prompt")
        return get_social_media_judge_prompt(prompt, model_output)

def call_judge_api(client, prompt, max_retries=3):
    """调用评测API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            
            # 尝试解析JSON结果
            try:
                # 移除可能的markdown代码块标记
                if result.startswith("```json"):
                    result = result[7:]
                if result.endswith("```"):
                    result = result[:-3]
                
                json_result = json.loads(result)
                return json_result
            except json.JSONDecodeError as e:
                print(f"JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"原始响应: {result}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
                continue
                
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    
    return None

def calculate_avg_score(eval_dict):
    """计算平均分"""
    scores = [v for k, v in eval_dict.items() if k.startswith('score_')]
    return sum(scores) / len(scores) if scores else 0

def main():
    """主函数，执行评测流程"""
    print("--- 单模型评测脚本启动 (使用硅基流动API) ---")

    # 检查依赖
    if not check_dependencies():
        return

    # 检查API密钥
    if SILICONFLOW_API_KEY == "sk-YOUR_API_KEY_HERE" or not SILICONFLOW_API_KEY:
        print("错误：请在.env文件中填入您的硅基流动API密钥。")
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

    # 加载数据
    evaluation_df, val_data = load_data()
    if evaluation_df is None or val_data is None:
        return

    # 检查数据长度是否匹配
    if len(evaluation_df) != len(val_data):
        print(f"错误：评测数据长度 ({len(evaluation_df)}) 与验证数据长度 ({len(val_data)}) 不匹配")
        return

    print(f"开始评测 {len(evaluation_df)} 条数据...")
    print(f"使用响应键: {RESPONSE_KEY}")
    print(f"输出文件: {OUTPUT_CSV_PATH}")

    # 存储评测结果
    judge_results = []

    # 遍历数据进行评测
    for idx in tqdm(range(len(evaluation_df)), desc="评测进度"):
        try:
            # 获取当前行的数据
            row = evaluation_df.iloc[idx]
            val_item = val_data[idx]
            
            # 提取数据
            prompt = row['prompt']
            model_output = row[RESPONSE_KEY]
            content_type = val_item['type']
            
            # 根据内容类型选择评测prompt
            judge_prompt = get_judge_prompt_by_type(
                content_type, 
                prompt, 
                model_output
            )
            
            # 调用评测API
            print(f"\n正在评测第 {idx + 1} 条数据 (类型: {content_type})...")
            result = call_judge_api(client, judge_prompt)
            
            if result is None:
                print(f"第 {idx + 1} 条数据评测失败，跳过")
                continue
            
            # 解析评测结果
            try:
                # 提取模型评分
                model_eval = result.get('model_evaluation', {})
                
                # 计算平均分
                avg_score = calculate_avg_score(model_eval)
                
                # 存储结果
                judge_result = {
                    'index': idx,
                    'content_type': content_type,
                    'avg_score': avg_score,
                    'critique': model_eval.get('critique', ''),
                    'scores': model_eval
                }
                
                judge_results.append(judge_result)
                
                print(f"评测完成 - 平均分: {avg_score:.2f}")
                
            except Exception as e:
                print(f"解析评测结果失败: {e}")
                continue
            
            # 避免API调用过于频繁
            time.sleep(1)
            
        except Exception as e:
            print(f"处理第 {idx + 1} 条数据时出错: {e}")
            continue

    # 保存评测结果
    if judge_results:
        print(f"\n正在保存评测结果到 {OUTPUT_CSV_PATH}...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(judge_results)
        
        # 保存到CSV
        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        
        # 生成统计报告
        print("\n=== 评测统计报告 ===")
        print(f"总评测数据: {len(judge_results)} 条")
        
        # 按内容类型统计
        if len(judge_results) > 0:
            type_stats = results_df.groupby('content_type').agg({
                'avg_score': ['count', 'mean']
            }).round(2)
            
            print("\n按内容类型统计:")
            print(type_stats)
            
            # 整体平均分
            overall_avg = results_df['avg_score'].mean()
            print(f"\n整体平均分: {overall_avg:.2f}")
            
            # 各维度平均分（如果有的话）
            score_columns = [col for col in results_df.columns if col.startswith('scores')]
            if score_columns:
                print("\n各维度平均分:")
                for col in score_columns:
                    if isinstance(results_df[col].iloc[0], dict):
                        # 如果是字典，提取分数
                        scores_data = []
                        for scores_dict in results_df[col]:
                            if isinstance(scores_dict, dict):
                                scores_data.append(scores_dict)
                        
                        if scores_data:
                            scores_df = pd.DataFrame(scores_data)
                            score_cols = [c for c in scores_df.columns if c.startswith('score_')]
                            if score_cols:
                                avg_scores = scores_df[score_cols].mean()
                                for score_name, avg_score in avg_scores.items():
                                    print(f"  {score_name}: {avg_score:.2f}")
        
        print(f"\n评测结果已保存到: {OUTPUT_CSV_PATH}")
    else:
        print("没有成功评测的数据")

if __name__ == "__main__":
    main()
