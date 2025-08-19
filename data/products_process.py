import pandas as pd
import os

def process_products_csv():
    """
    读取products.csv文件，将字段中的|和空格替换为逗号
    """
    # 文件路径
    csv_file = './data/products.csv'
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误：找不到文件 {csv_file}")
        return
    
    try:
        # 读取CSV文件
        print(f"正在读取 {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"成功读取文件，共 {len(df)} 行数据")
        print(f"列名：{list(df.columns)}")
        
        # 处理每一列，将|和空格替换为逗号
        for column in df.columns:
            if df[column].dtype == 'object':  # 只处理字符串类型的列
                print(f"正在处理列：{column}")
                # 先替换竖线，再处理连续空格
                df[column] = df[column].str.replace('|', ', ')
                # 处理连续空格（2个或更多空格）
                df[column] = df[column].str.replace(r'\s{2,}', ', ', regex=True)
                # 处理单个空格（在特定情况下）
                df[column] = df[column].str.replace(r'(\w+)\s+(\w+)', r'\1, \2', regex=True)
        
        # 特别处理target_audience列，确保格式正确
        if 'target_audience' in df.columns:
            print(f"\n特别处理target_audience列...")
            # 显示处理前后的对比
            print(f"处理前target_audience列示例：")
            print(df['target_audience'].head(3))
            
            # 清理多余的空格和逗号
            df['target_audience'] = df['target_audience'].str.strip()
            df['target_audience'] = df['target_audience'].str.replace(r',\s*,', ',', regex=True)  # 清理多余的逗号
            df['target_audience'] = df['target_audience'].str.replace(r'^\s*,\s*|\s*,\s*$', '', regex=True)  # 清理开头和结尾的逗号
            
            print(f"处理后target_audience列示例：")
            print(df['target_audience'].head(3))
        
        # 保存处理后的文件
        output_file = './data/products_processed.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n处理完成！")
        print(f"原始文件：{csv_file}")
        print(f"处理后文件：{output_file}")
        
        # 显示前几行数据作为示例
        print(f"\n处理后的前3行数据：")
        print(df.head(3).to_string())
        
        # 显示每列的数据类型
        print(f"\n各列数据类型：")
        print(df.dtypes)
        
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

def main():
    """主函数"""
    print("=" * 50)
    print("原素方程产品数据处理工具")
    print("=" * 50)
    
    # 处理CSV文件
    process_products_csv()
    
    print("\n" + "=" * 50)
    print("处理完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
