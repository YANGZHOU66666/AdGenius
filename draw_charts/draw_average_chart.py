import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_average_chart():
    """
    绘制两个模型（Qwen2.5-7B 和 Qwen3-4B）在不同版本（base、sft、grpo）下的能力总分对比柱状图
    """
    
    # 定义文件路径和对应的模型版本
    files = {
        'Qwen2.5-7B-Instruct': {
            'base': 'results/judge_new_Qwen2_5_7b_Instruct_model_results.csv',
            'sft': 'results/judge_new_Qwen2_5_7b_Instruct_sft_model_results.csv', 
            'grpo': 'results/judge_new_Qwen2_5_7b_Instruct_grpo_model_results.csv'
        },
        'Qwen3-4B-Instruct-2507': {
            'base': 'results/judge_new_Qwen3_4b_model_results.csv',
            'sft': 'results/judge_new_Qwen3_4b_Instruct_sft_model_results.csv',
            'grpo': 'results/judge_new_Qwen3_4b_Instruct_grpo_model_results.csv'
        }
    }
    
    # 存储每个模型版本的平均分
    model_scores = {}
    
    # 读取每个文件并计算平均分
    for model_name, versions in files.items():
        model_scores[model_name] = {}
        for version, file_path in versions.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                avg_score = df['avg_score'].mean()
                model_scores[model_name][version] = avg_score
                print(f"{model_name} {version}: 平均分 {avg_score:.3f}")
            else:
                print(f"警告：文件 {file_path} 不存在")
                model_scores[model_name][version] = 0
    
    # 准备绘图数据 - 按模型分组，同一基模的三个版本连在一起
    models = list(model_scores.keys())
    versions = ['base', 'sft', 'grpo']
    version_labels = ['Base', 'SFT', 'GRPO']
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 设置柱状图参数
    # 为每个基模分配一个位置，同一基模的三个版本有少量间隔
    group_positions = [0, 4]  # 两个基模的位置
    width = 0.6  # 每个柱子宽度
    version_spacing = 0.1  # 同一基模内版本之间的间隔
    
    # 为每个模型和版本组合创建柱子
    # 定义版本颜色方案，相同版本使用相同颜色
    version_colors = {
        'base': '#a1c4fd',   # 浅蓝色
        'sft': '#7bb3f0',    # 中蓝色  
        'grpo': '#4a90e2'    # 深蓝色
    }
    
    bar_heights = []
    bar_positions = []
    
    for i, model in enumerate(models):
        group_center = group_positions[i]
        for j, version in enumerate(versions):
            # 同一基模的三个版本有少量间隔，从中心向两边分布
            x_pos = group_center + (j - 1) * (width + version_spacing)  # 添加版本间隔
            score = model_scores[model][version]
            bar_heights.append(score)
            bar_positions.append(x_pos)
            ax.bar(x_pos, score, width, color=version_colors[version], alpha=0.8)
    
    # 设置x轴标签 - 只在每个组的中心位置显示模型名
    ax.set_xticks(group_positions)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    
    # 在柱状图上添加数值标签
    for i, (pos, height) in enumerate(zip(bar_positions, bar_heights)):
        ax.text(pos, height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 创建图例 - 显示版本标签
    legend_elements = []
    for version, color in version_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=version.upper()))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             title='模型版本', title_fontsize=12, frameon=True)
    
    # 设置图表属性
    ax.set_xlabel('基模', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均能力总分', fontsize=12, fontweight='bold')
    ax.set_title('Base, SFT, GRPO模型能力对比', fontsize=14, fontweight='bold', pad=20)
    # 移除网格线
    
    # 设置y轴范围
    all_scores = [score for model_scores in model_scores.values() for score in model_scores.values()]
    min_score = min(all_scores) - 0.1
    max_score = max(all_scores) + 0.1
    ax.set_ylim(min_score, max_score)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = 'draw_charts/model_comparison_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印详细统计信息
    print("\n=== 详细统计信息 ===")
    for model_name, versions in model_scores.items():
        print(f"\n{model_name}:")
        for version, score in versions.items():
            print(f"  {version.upper()}: {score:.3f}")
    
    # 计算改进幅度
    print("\n=== 改进幅度分析 ===")
    for model_name, versions in model_scores.items():
        base_score = versions['base']
        sft_improvement = versions['sft'] - base_score
        grpo_improvement = versions['grpo'] - base_score
        print(f"\n{model_name}:")
        print(f"  SFT相对Base改进: {sft_improvement:+.3f}")
        print(f"  GRPO相对Base改进: {grpo_improvement:+.3f}")
        print(f"  GRPO相对SFT改进: {grpo_improvement - sft_improvement:+.3f}")

if __name__ == "__main__":
    # 确保在项目根目录中运行
    if not os.path.exists("results"):
        print("错误：请在项目根目录中运行此脚本")
        exit(1)
    
    draw_average_chart()
