# 将rm_dataset.csv的chosen列更改为添加CTA后的数据
import json
import pandas as pd
import os

def load_sft_data_cta(file_path: str):
    """加载sft_dataset_cleaned_cta.jsonl文件，提取output列"""
    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                outputs.append(data["output"])
    return outputs

def update_rm_dataset(rm_csv_path: str, sft_cta_path: str, output_path: str):
    """更新rm_dataset.csv，将chosen列替换为sft_dataset_cleaned_cta.jsonl的output"""
    
    print("开始更新rm_dataset.csv...")
    
    # 检查输入文件是否存在
    if not os.path.exists(rm_csv_path):
        print(f"错误：找不到rm_dataset.csv文件: {rm_csv_path}")
        return
    
    if not os.path.exists(sft_cta_path):
        print(f"错误：找不到sft_dataset_cleaned_cta.jsonl文件: {sft_cta_path}")
        return
    
    # 加载rm_dataset.csv
    print(f"正在加载rm_dataset.csv: {rm_csv_path}")
    try:
        rm_df = pd.read_csv(rm_csv_path)
        print(f"成功加载rm_dataset.csv，共 {len(rm_df)} 行数据")
    except Exception as e:
        print(f"加载rm_dataset.csv失败: {e}")
        return
    
    # 加载sft_dataset_cleaned_cta.jsonl的output
    print(f"正在加载sft_dataset_cleaned_cta.jsonl: {sft_cta_path}")
    try:
        sft_outputs = load_sft_data_cta(sft_cta_path)
        print(f"成功加载sft_dataset_cleaned_cta.jsonl，共 {len(sft_outputs)} 行数据")
    except Exception as e:
        print(f"加载sft_dataset_cleaned_cta.jsonl失败: {e}")
        return
    
    # 检查数据条数是否一致
    if len(rm_df) != len(sft_outputs):
        print(f"错误：数据条数不一致！rm_dataset.csv: {len(rm_df)}, sft_dataset_cleaned_cta.jsonl: {len(sft_outputs)}")
        return
    
    print("数据条数验证通过，开始替换chosen列...")
    
    # 创建新的DataFrame，只包含需要的列
    updated_df = pd.DataFrame({
        'prompt': rm_df['prompt'],
        'chosen': sft_outputs,
        'rejected': rm_df['rejected'],
        'method': rm_df['method']
    })
    
    # 保存更新后的数据
    print(f"正在保存更新后的数据到: {output_path}")
    try:
        updated_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"成功保存更新后的数据到: {output_path}")
    except Exception as e:
        print(f"保存数据失败: {e}")
        return
    
    # 显示更新统计
    print(f"\n=== 更新完成 ===")
    print(f"原始数据行数: {len(rm_df)}")
    print(f"更新后数据行数: {len(updated_df)}")
    print(f"chosen列已更新为sft_dataset_cleaned_cta.jsonl的output")
    print(f"输出文件: {output_path}")
    
    # 显示前几行数据作为预览
    print(f"\n数据预览:")
    print(updated_df.head())
    
    # 显示列信息
    print(f"\n列信息:")
    print(f"列数: {len(updated_df.columns)}")
    print(f"列名: {list(updated_df.columns)}")

def main():
    """主函数"""
    print("=== 开始更新rm_dataset.csv的chosen列 ===")
    
    # 文件路径
    rm_csv_path = "data/rm_dataset.csv"
    sft_cta_path = "data/sft_dataset_cleaned_cta.jsonl"
    output_path = "data/rm_dataset_updated.csv"
    
    # 执行更新
    update_rm_dataset(rm_csv_path, sft_cta_path, output_path)
    
    print("\n=== 脚本执行完成 ===")

if __name__ == "__main__":
    main()
