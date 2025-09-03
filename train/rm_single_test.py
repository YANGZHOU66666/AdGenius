import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_reward_model(model_path: str):
    """
    加载奖励模型和分词器
    
    Args:
        model_path: 模型路径
        
    Returns:
        model, tokenizer, device
    """
    print(f"--- 正在加载奖励模型: {model_path} ---")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        
        # 检查是否有可用的 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()  # 设置为评估模式
        
        print(f"--- 模型加载成功，运行在: {device} ---")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None, None


def calculate_reward_score(model, tokenizer, device, prompt: str, response: str) -> float:
    """
    计算单个回答的奖励分数
    
    Args:
        model: 奖励模型
        tokenizer: 分词器
        device: 设备
        prompt: 用户提示
        response: 模型回答
        
    Returns:
        奖励分数
    """
    # 构建对话消息
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # 应用聊天模板并转换为tensor
    try:
        input_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        # 如果聊天模板失败，使用简单格式
        input_text = tokenizer(
            f"用户: {prompt}\n助手: {response}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
    
    # 模型推理
    with torch.no_grad():
        score = model(input_text).logits[0].item()
    
    return score


def test_single_prompt_response():
    """
    测试单条 prompt-response 对
    """
    print("=== 单条奖励模型评分测试 ===")
    
    # --- 配置 ---
    model_path = "models/Skywork-Reward-V2-Qwen3-0.6B-finetuned"  # 奖励模型路径
    
    # --- 测试用例 ---
    prompt = "请为\"原素方程\"品牌的如下产品创作一条侧重使用反馈的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}"
    
    response = """护肤小课堂 | 熬夜党如何告别暗沉？解密玫瑰提取物的科学焕亮力

💡 问题：
经常被夸「气色不好」？护肤品用了一堆，肤色依然暗沉不均？可能你忽略了源头抗氧化这一步——当皮肤长期暴露在氧化压力下（紫外线/熬夜/污染），自由基会加速黑色素生成，就像切开的苹果迅速变黄。

🔬 成分科普：
真正的「抗氧焕亮高手」是玫瑰提取物（Rosa Damascena），不同于普通植物成分，它含有：
• 高浓度花青素（比蓝莓高3倍）——中和自由基链式反应
• 天然有机酸——温和代谢老废角质层
• 多糖复合物——同步修护肌肤屏障
（实验室数据：0.5%浓度使用28天，酪氨酸酶活性降低42%）

⚠️ 避坑指南：
1️⃣ 警惕「玫瑰水」概念添加（成分表排名垫底=无效）
2️⃣ 优先选CO2超临界萃取技术（保留活性物质更完整）
3️⃣ 与VC/烟酰胺复配可加乘美白通路

🌸 案例参考：
原素方程「玫瑰焕采精华」正是科学配方的典范：
✓ 5%玫瑰提取物（足量添加）+ 3%乙基VC（稳定衍生物）
✓ 积雪草提取物缓冲刺激（敏肌友好）
✓ 零酒精/色素（通过斑贴测试）

📝 我的使用反馈：
连续使用两周后，颧骨处的晒斑有肉眼可见的淡化（配合防晒），整体肤色的通透感提升明显——这种「从肌底透出来的光泽」正是科学配方的魅力。

#护肤成分 #科学护肤 #精简护肤 #原素方程 #抗氧化精华"""
    
    # 加载模型
    model, tokenizer, device = load_reward_model(model_path)
    if model is None:
        print("模型加载失败，测试终止")
        return
    
    # 计算分数
    print(f"\n--- 开始评分 ---")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Response: {response[:100]}...")
    
    score = calculate_reward_score(model, tokenizer, device, prompt, response)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("评分结果")
    print(f"{'='*60}")
    print(f"奖励分数: {score:.4f}")
    
    print("=== 评分测试完成 ===")


if __name__ == "__main__":
    try:
        test_single_prompt_response()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
