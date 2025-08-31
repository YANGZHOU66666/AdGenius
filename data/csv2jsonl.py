import pandas as pd
import json
import os
import sys
from typing import Dict, List

def csv_to_jsonl(csv_path: str, jsonl_path: str, encoding: str = 'utf-8'):
    """
    将CSV文件转换为JSONL格式
    
    Args:
        csv_path: 输入的CSV文件路径
        jsonl_path: 输出的JSONL文件路径
        encoding: CSV文件的编码格式，默认utf-8
    """
    
    print(f"开始转换CSV文件: {csv_path}")
    
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"成功读取CSV文件，共 {len(df)} 行，{len(df.columns)} 列")
        
        # 显示列信息
        print(f"列名: {list(df.columns)}")
        print(f"数据类型:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(jsonl_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        
        # 转换为JSONL格式
        print("正在转换为JSONL格式...")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                # 将每行转换为字典
                row_dict = row.to_dict()
                
                # 处理NaN值，转换为None
                for key, value in row_dict.items():
                    if pd.isna(value):
                        row_dict[key] = None
                
                # 写入JSONL文件
                json_line = json.dumps(row_dict, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"成功转换为JSONL格式，已保存到: {jsonl_path}")
        
        # 显示转换统计
        print(f"\n=== 转换完成 ===")
        print(f"输入文件: {csv_path}")
        print(f"输出文件: {jsonl_path}")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        
        # 显示前几行数据作为预览
        print(f"\n数据预览:")
        for i in range(min(3, len(df))):
            row_dict = df.iloc[i].to_dict()
            # 处理NaN值
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None
            print(f"第{i+1}行: {json.dumps(row_dict, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False
    
    return True

def batch_convert_csv_to_jsonl(input_dir: str, output_dir: str, encoding: str = 'utf-8'):
    """
    批量转换目录下的所有CSV文件为JSONL格式
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        encoding: CSV文件的编码格式
    """
    
    print(f"开始批量转换目录: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        return
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 查找所有CSV文件
    csv_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.csv'):
            csv_files.append(file)
    
    if not csv_files:
        print(f"在目录 {input_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 逐个转换
    success_count = 0
    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        jsonl_file = csv_file.replace('.csv', '.jsonl')
        jsonl_path = os.path.join(output_dir, jsonl_file)
        
        print(f"\n正在转换: {csv_file}")
        if csv_to_jsonl(csv_path, jsonl_path, encoding):
            success_count += 1
        else:
            print(f"转换失败: {csv_file}")
    
    print(f"\n=== 批量转换完成 ===")
    print(f"成功转换: {success_count}/{len(csv_files)} 个文件")
    print(f"输出目录: {output_dir}")

def main():
    """主函数"""
    print("=== CSV转JSONL工具 ===")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) >= 3:
            csv_path = sys.argv[1]
            jsonl_path = sys.argv[2]
            encoding = sys.argv[3] if len(sys.argv) > 3 else 'utf-8'
            
            csv_to_jsonl(csv_path, jsonl_path, encoding)
        else:
            print("用法: python csv2jsonl.py <csv文件路径> <jsonl文件路径> [编码格式]")
            print("示例: python csv2jsonl.py data.csv data.jsonl utf-8")
    else:
        # 交互模式
        print("请选择转换模式:")
        print("1. 转换单个CSV文件")
        print("2. 批量转换目录下的CSV文件")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            # 单个文件转换
            csv_path = input("请输入CSV文件路径: ").strip()
            jsonl_path = input("请输入输出JSONL文件路径: ").strip()
            encoding = input("请输入CSV文件编码格式 (默认utf-8): ").strip() or 'utf-8'
            
            csv_to_jsonl(csv_path, jsonl_path, encoding)
            
        elif choice == "2":
            # 批量转换
            input_dir = input("请输入输入目录路径: ").strip()
            output_dir = input("请输入输出目录路径: ").strip()
            encoding = input("请输入CSV文件编码格式 (默认utf-8): ").strip() or 'utf-8'
            
            batch_convert_csv_to_jsonl(input_dir, output_dir, encoding)
            
        else:
            print("无效选择")

if __name__ == "__main__":
    main()
