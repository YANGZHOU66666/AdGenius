import pandas as pd
import os

def merge_results():
    """
    合并 evaluation_responses_final_Qwen3.csv 和 Qwen3_4B_grpo_responses.csv
    生成新的 Qwen3_4B_all_responses.csv 文件
    """
    
    # 读取第一个文件
    file1_path = "responses/evaluation_responses_final_Qwen3.csv"
    file2_path = "responses/Qwen3_4B_grpo_responses.csv"
    output_path = "responses/Qwen3_4B_all_responses.csv"
    
    print(f"正在读取 {file1_path}...")
    df1 = pd.read_csv(file1_path)
    print(f"文件1包含 {len(df1)} 行数据")
    print(f"文件1列名: {list(df1.columns)}")
    
    print(f"\n正在读取 {file2_path}...")
    df2 = pd.read_csv(file2_path)
    print(f"文件2包含 {len(df2)} 行数据")
    print(f"文件2列名: {list(df2.columns)}")
    
    # 从第一个文件中选择需要的列
    df1_selected = df1[['prompt', 'base_model_output', 'finetuned_model_output']].copy()
    
    # 重命名列
    df1_selected = df1_selected.rename(columns={
        'finetuned_model_output': 'sft_model_output'
    })
    
    # 从第二个文件中选择需要的列
    df2_selected = df2[['prompt', 'model_output']].copy()
    
    # 重命名列
    df2_selected = df2_selected.rename(columns={
        'model_output': 'grpo_model_output'
    })
    
    print(f"\n合并前数据预览:")
    print("文件1选择列后:")
    print(df1_selected.head(2))
    print("\n文件2选择列后:")
    print(df2_selected.head(2))
    
    # 基于prompt列进行合并
    print(f"\n正在基于prompt列合并数据...")
    merged_df = pd.merge(df1_selected, df2_selected, on='prompt', how='inner')
    
    print(f"合并后包含 {len(merged_df)} 行数据")
    print(f"合并后列名: {list(merged_df.columns)}")
    
    # 保存合并后的数据
    print(f"\n正在保存到 {output_path}...")
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ 合并完成！输出文件: {output_path}")
    print(f"最终数据包含 {len(merged_df)} 行，{len(merged_df.columns)} 列")
    
    # 显示最终数据的预览
    print(f"\n最终数据预览:")
    print(merged_df.head(3))
    
    return merged_df

if __name__ == "__main__":
    # 确保在项目根目录中运行
    if not os.path.exists("responses/evaluation_responses_final_Qwen3.csv"):
        print("错误：请在项目根目录中运行此脚本")
        exit(1)
    
    merged_data = merge_results()
