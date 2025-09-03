import pandas as pd
import json
import os
from typing import List, Dict, Any


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    加载CSV文件数据
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        pandas DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")
    
    print(f"正在加载CSV文件: {file_path}")
    df = pd.read_csv(file_path)
    print(f"成功加载CSV数据: {len(df)} 条")
    
    return df


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件数据
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含JSON数据的列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL文件不存在: {file_path}")
    
    print(f"正在加载JSONL文件: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行JSON解析失败: {e}")
                    continue
    
    print(f"成功加载JSONL数据: {len(data)} 条")
    return data


def merge_data(csv_df: pd.DataFrame, jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并CSV和JSONL数据
    
    Args:
        csv_df: CSV数据DataFrame
        jsonl_data: JSONL数据列表
        
    Returns:
        合并后的数据列表
    """
    print("正在合并数据...")
    
    # 检查数据长度是否匹配
    if len(csv_df) != len(jsonl_data):
        print(f"警告：CSV数据长度 ({len(csv_df)}) 与JSONL数据长度 ({len(jsonl_data)}) 不匹配")
        print("将按较短的长度进行合并")
        min_length = min(len(csv_df), len(jsonl_data))
        csv_df = csv_df.head(min_length)
        jsonl_data = jsonl_data[:min_length]
    
    merged_data = []
    
    for i in range(len(csv_df)):
        try:
            # 从CSV获取rejected和method
            csv_row = csv_df.iloc[i]
            rejected = csv_row['rejected']
            method = csv_row['method']
            
            # 从JSONL获取prompt, chosen, type
            jsonl_item = jsonl_data[i]
            prompt = jsonl_item['prompt']
            chosen = jsonl_item['chosen']
            content_type = jsonl_item['type']
            
            # 合并数据
            merged_item = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'method': method,
                'type': content_type
            }
            
            merged_data.append(merged_item)
            
        except Exception as e:
            print(f"警告：合并第 {i+1} 条数据时出错: {e}")
            continue
    
    print(f"成功合并数据: {len(merged_data)} 条")
    return merged_data


def save_jsonl_data(data: List[Dict[str, Any]], output_path: str):
    """
    保存数据到JSONL文件
    
    Args:
        data: 要保存的数据列表
        output_path: 输出文件路径
    """
    print(f"正在保存数据到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"成功保存 {len(data)} 条数据到 {output_path}")


def main():
    """
    主函数，执行数据合并流程
    """
    print("=== 开始合并验证数据集 ===")
    
    # 文件路径配置
    csv_file_path = "data/val_rm_dataset_temp.csv"
    jsonl_file_path = "data/val_dataset_final_deprecated.jsonl"
    output_file_path = "data/val_dataset_final.jsonl"
    
    try:
        # 加载数据
        csv_df = load_csv_data(csv_file_path)
        jsonl_data = load_jsonl_data(jsonl_file_path)
        
        # 检查CSV文件是否包含必要的列
        required_csv_columns = ['rejected', 'method']
        missing_columns = [col for col in required_csv_columns if col not in csv_df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        # 检查JSONL数据是否包含必要的键
        if jsonl_data:
            sample_item = jsonl_data[0]
            required_jsonl_keys = ['prompt', 'chosen', 'type']
            missing_keys = [key for key in required_jsonl_keys if key not in sample_item]
            if missing_keys:
                raise ValueError(f"JSONL文件缺少必要的键: {missing_keys}")
        
        # 合并数据
        merged_data = merge_data(csv_df, jsonl_data)
        
        if not merged_data:
            print("错误：没有成功合并的数据")
            return
        
        # 保存合并后的数据
        save_jsonl_data(merged_data, output_file_path)
        
        # 显示统计信息
        print("\n=== 合并统计信息 ===")
        print(f"CSV数据条数: {len(csv_df)}")
        print(f"JSONL数据条数: {len(jsonl_data)}")
        print(f"合并后数据条数: {len(merged_data)}")
        
        # 按内容类型统计
        type_counts = {}
        for item in merged_data:
            content_type = item.get('type', 'unknown')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        print("\n按内容类型统计:")
        for content_type, count in type_counts.items():
            print(f"  {content_type}: {count} 条")
        
        # 按方法统计
        method_counts = {}
        for item in merged_data:
            method = item.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("\n按方法统计:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} 条")
        
        print(f"\n=== 合并完成 ===")
        print(f"输出文件: {output_file_path}")
        
    except Exception as e:
        print(f"合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
